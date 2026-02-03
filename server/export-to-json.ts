// Export script: Transform Shopify products and generate embeddings, then export to JSON for Compass import
import Database from "better-sqlite3"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import path from "path"
import fs from "fs"
import "dotenv/config"

// SQLite product structure
interface SQLiteProduct {
  sku: string
  handle: string
  title: string
  description: string
  vendor: string
  price: number
  currency: string
  image_url: string
  product_url: string
  tags: string
  search_content: string
}

// MongoDB document structure
interface MongoProduct {
  sku: string
  handle: string
  title: string
  description: string
  vendor: string
  price: number
  currency: string
  image_url: string
  product_url: string
  tags: string[]
  search_content: string
  embedding_text: string
  embedding?: number[]
}

// Retry with exponential backoff for rate limits
async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries = 3
): Promise<T> {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await fn()
    } catch (error: any) {
      if (error.status === 429 && attempt < maxRetries) {
        const delay = Math.min(1000 * Math.pow(2, attempt), 30000)
        console.log(`Rate limit hit. Retrying in ${delay / 1000} seconds...`)
        await new Promise((resolve) => setTimeout(resolve, delay))
        continue
      }
      throw error
    }
  }
  throw new Error("Max retries exceeded")
}

// Extract products from SQLite
function extractProductsFromSQLite(dbPath: string): SQLiteProduct[] {
  console.log(`Opening SQLite database: ${dbPath}`)
  const db = new Database(dbPath, { readonly: true })

  try {
    const query = `SELECT sku, handle, title, description, vendor, price, currency, 
                          image_url, product_url, tags, search_content 
                   FROM products`
    const products = db.prepare(query).all() as SQLiteProduct[]
    console.log(`Extracted ${products.length} products from SQLite`)
    return products
  } finally {
    db.close()
  }
}

// Transform product schema
function transformProduct(sqliteProduct: SQLiteProduct): MongoProduct {
  let tags: string[] = []
  if (sqliteProduct.tags) {
    try {
      tags = JSON.parse(sqliteProduct.tags)
      if (!Array.isArray(tags)) tags = [String(tags)]
    } catch {
      tags = sqliteProduct.tags.split(",").map((t) => t.trim()).filter((t) => t.length > 0)
    }
  }

  const embedding_text = generateEmbeddingText(sqliteProduct, tags)

  return {
    sku: sqliteProduct.sku,
    handle: sqliteProduct.handle,
    title: sqliteProduct.title,
    description: sqliteProduct.description || "",
    vendor: sqliteProduct.vendor,
    price: sqliteProduct.price,
    currency: sqliteProduct.currency,
    image_url: sqliteProduct.image_url,
    product_url: sqliteProduct.product_url,
    tags: tags,
    search_content: sqliteProduct.search_content || "",
    embedding_text: embedding_text,
  }
}

function generateEmbeddingText(product: SQLiteProduct, tags: string[]): string {
  const basicInfo = `${product.title} ${product.description || ""} from ${product.vendor}`
  const pricing = `Price: ${product.price} ${product.currency}`
  const tagText = tags.length > 0 ? `Tags: ${tags.join(", ")}` : ""
  const searchContent = product.search_content || ""
  return `${basicInfo}. ${pricing}. ${tagText}. ${searchContent}`.trim()
}

// Main export function
async function exportToJson(): Promise<void> {
  const startTime = Date.now()
  console.log("🚀 Starting Shopify product export with embeddings...\n")

  const BATCH_SIZE = 25
  const BATCH_DELAY = 1500
  const sqliteDbPath = path.join(__dirname, "..", "easymart.db")
  const outputPath = path.join(__dirname, "products-for-import.json")

  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY environment variable is required")
  }

  try {
    // Step 1: Extract from SQLite
    console.log("📦 Step 1: Extracting products from SQLite...")
    const sqliteProducts = extractProductsFromSQLite(sqliteDbPath)

    if (sqliteProducts.length === 0) {
      console.log("⚠ No products found. Exiting.")
      return
    }

    // Step 2: Transform products
    console.log("\n🔄 Step 2: Transforming product schema...")
    const mongoProducts = sqliteProducts.map(transformProduct)
    console.log(`Transformed ${mongoProducts.length} products`)

    // Step 3: Initialize embeddings model
    console.log("\n🧠 Step 3: Initializing AI embeddings model...")
    const embeddingsModel = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "text-embedding-004",
    })
    console.log("✓ Embeddings model ready")

    // Step 4: Generate embeddings in batches
    console.log(`\n⚙️  Step 4: Generating embeddings for ${mongoProducts.length} products...`)
    const productsWithEmbeddings: MongoProduct[] = []
    let successCount = 0
    let failCount = 0

    for (let i = 0; i < mongoProducts.length; i += BATCH_SIZE) {
      const batch = mongoProducts.slice(i, i + BATCH_SIZE)
      const batchNum = Math.floor(i / BATCH_SIZE) + 1
      const totalBatches = Math.ceil(mongoProducts.length / BATCH_SIZE)

      console.log(`\n📦 Batch ${batchNum}/${totalBatches} (${batch.length} products)`)

      for (const product of batch) {
        try {
          const embedding = await retryWithBackoff(async () => {
            return await embeddingsModel.embedQuery(product.embedding_text)
          })

          productsWithEmbeddings.push({ ...product, embedding })
          successCount++
          
          // Progress indicator every 10 products
          if (successCount % 10 === 0) {
            console.log(`   ✓ Processed ${successCount}/${mongoProducts.length} products`)
          }
        } catch (error: any) {
          failCount++
          console.error(`   ✗ Failed: ${product.sku} - ${error.message}`)
          // Still add without embedding for manual fix later
          productsWithEmbeddings.push(product)
        }
      }

      // Delay between batches
      if (i + BATCH_SIZE < mongoProducts.length) {
        console.log(`   ⏳ Waiting ${BATCH_DELAY / 1000}s before next batch...`)
        await new Promise((resolve) => setTimeout(resolve, BATCH_DELAY))
      }
    }

    // Step 5: Export to JSON
    console.log("\n💾 Step 5: Exporting to JSON file...")
    fs.writeFileSync(outputPath, JSON.stringify(productsWithEmbeddings, null, 2))
    console.log(`✓ Exported to: ${outputPath}`)

    // Final report
    const duration = ((Date.now() - startTime) / 1000).toFixed(2)
    const fileSizeMB = (fs.statSync(outputPath).size / (1024 * 1024)).toFixed(2)

    console.log(`\n${"=".repeat(60)}`)
    console.log("🎉 EXPORT COMPLETE!")
    console.log(`${"=".repeat(60)}`)
    console.log(`✓ Successfully processed: ${successCount} products`)
    console.log(`✗ Failed embeddings: ${failCount} products`)
    console.log(`📁 Output file: ${outputPath}`)
    console.log(`📊 File size: ${fileSizeMB} MB`)
    console.log(`⏱️  Total time: ${duration}s`)
    console.log(`${"=".repeat(60)}`)
    console.log(`\n📋 NEXT STEPS:`)
    console.log(`1. Open MongoDB Compass`)
    console.log(`2. Click on "inventory_database" (create it if needed)`)
    console.log(`3. Click on "items" collection (create it if needed)`)
    console.log(`4. Click "Add Data" → "Import JSON or CSV file"`)
    console.log(`5. Select: ${outputPath}`)
    console.log(`6. Click "Import"\n`)

  } catch (error) {
    console.error("\n❌ Export failed:", error)
    throw error
  }
}

// Execute
exportToJson().catch((error) => {
  console.error("Fatal error:", error)
  process.exit(1)
})
