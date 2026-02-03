// Migration script to transfer Shopify products from SQLite to MongoDB with vector embeddings
import Database from "better-sqlite3"
import { MongoClient } from "mongodb"
import { GoogleGenerativeAIEmbeddings } from "@langchain/google-genai"
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb"
import path from "path"
import "dotenv/config"

// SQLite product structure from easymart.db
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
  tags: string // JSON string or comma-separated
  search_content: string
}

// MongoDB document structure for products
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
  embedding_text: string // Comprehensive searchable summary
  embedding?: number[] // 768-dimensional vector (added during embedding generation)
}

// Utility function to handle API rate limits with exponential backoff
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

// Extract products from SQLite database
function extractProductsFromSQLite(dbPath: string): SQLiteProduct[] {
  console.log(`Opening SQLite database: ${dbPath}`)
  const db = new Database(dbPath, { readonly: true })

  try {
    // Query all products from the database
    const query = `SELECT sku, handle, title, description, vendor, price, currency, 
                          image_url, product_url, tags, search_content 
                   FROM products`
    const products = db.prepare(query).all() as SQLiteProduct[]

    console.log(`Extracted ${products.length} products from SQLite`)
    return products
  } catch (error) {
    console.error("Error reading from SQLite:", error)
    throw error
  } finally {
    db.close()
  }
}

// Transform SQLite product to MongoDB schema
function transformProductToMongoSchema(sqliteProduct: SQLiteProduct): MongoProduct {
  // Parse tags - handle both JSON array and comma-separated strings
  let tags: string[] = []
  if (sqliteProduct.tags) {
    try {
      // Try parsing as JSON first
      tags = JSON.parse(sqliteProduct.tags)
      if (!Array.isArray(tags)) {
        // If parsed but not an array, convert to array
        tags = [String(tags)]
      }
    } catch {
      // If JSON parse fails, treat as comma-separated string
      tags = sqliteProduct.tags
        .split(",")
        .map((tag) => tag.trim())
        .filter((tag) => tag.length > 0)
    }
  }

  // Generate comprehensive searchable text summary
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

// Generate comprehensive text for embedding and search
function generateEmbeddingText(product: SQLiteProduct, tags: string[]): string {
  const basicInfo = `${product.title} ${product.description || ""} from ${product.vendor}`
  const pricing = `Price: ${product.price} ${product.currency}`
  const tagText = tags.length > 0 ? `Tags: ${tags.join(", ")}` : ""
  const searchContent = product.search_content || ""

  return `${basicInfo}. ${pricing}. ${tagText}. ${searchContent}`.trim()
}

// Process products in batches to generate embeddings
async function generateEmbeddingsForBatch(
  products: MongoProduct[],
  embeddings: GoogleGenerativeAIEmbeddings
): Promise<MongoProduct[]> {
  const productsWithEmbeddings: MongoProduct[] = []

  for (const product of products) {
    try {
      console.log(`Generating embedding for: ${product.sku} - ${product.title}`)

      // Generate embedding with retry logic
      const embedding = await retryWithBackoff(async () => {
        return await embeddings.embedQuery(product.embedding_text)
      })

      productsWithEmbeddings.push({
        ...product,
        embedding: embedding,
      })

      console.log(`✓ Successfully generated embedding for ${product.sku}`)
    } catch (error) {
      console.error(`✗ Failed to generate embedding for ${product.sku}:`, error)
      // Continue with next product instead of failing entire batch
    }
  }

  return productsWithEmbeddings
}

// Create vector search index if it doesn't exist
async function ensureVectorSearchIndex(client: MongoClient): Promise<void> {
  try {
    const db = client.db("inventory_database")
    const collection = db.collection("items")

    // Check if index already exists
    const indexes = await collection.listSearchIndexes().toArray()
    const vectorIndexExists = indexes.some((idx: any) => idx.name === "vector_index")

    if (vectorIndexExists) {
      console.log("Vector search index already exists")
      return
    }

    console.log("Creating vector search index...")
    const vectorSearchIdx = {
      name: "vector_index",
      type: "vectorSearch",
      definition: {
        fields: [
          {
            type: "vector",
            path: "embedding",
            numDimensions: 768,
            similarity: "cosine",
          },
        ],
      },
    }

    await collection.createSearchIndex(vectorSearchIdx)
    console.log("Successfully created vector search index")
  } catch (error) {
    console.error("Error managing vector search index:", error)
    // Don't throw - index might already exist or need to be created via Atlas UI
    console.log("Note: If index creation failed, you may need to create it via MongoDB Atlas UI")
  }
}

// Insert products into MongoDB
async function insertProductsToMongo(
  client: MongoClient,
  products: MongoProduct[]
): Promise<{ success: number; failed: number }> {
  const db = client.db("inventory_database")
  const collection = db.collection("items")

  let success = 0
  let failed = 0

  console.log(`\nInserting ${products.length} products into MongoDB...`)

  for (const product of products) {
    try {
      await collection.insertOne(product)
      success++
      console.log(`✓ Inserted: ${product.sku} - ${product.title}`)
    } catch (error: any) {
      failed++
      if (error.code === 11000) {
        console.log(`⚠ Duplicate SKU (skipped): ${product.sku}`)
      } else {
        console.error(`✗ Failed to insert ${product.sku}:`, error.message)
      }
    }
  }

  return { success, failed }
}

// Verify migration results
async function verifyMigration(client: MongoClient): Promise<void> {
  const db = client.db("inventory_database")
  const collection = db.collection("items")

  const totalCount = await collection.countDocuments()
  const withEmbeddings = await collection.countDocuments({
    embedding: { $exists: true, $ne: null },
  })

  console.log(`\n📊 Migration Verification:`)
  console.log(`   Total documents: ${totalCount}`)
  console.log(`   With embeddings: ${withEmbeddings}`)
  console.log(`   Missing embeddings: ${totalCount - withEmbeddings}`)

  // Get a sample product
  const sample = await collection.findOne({})
  if (sample) {
    console.log(`\n📄 Sample product:`)
    console.log(`   SKU: ${sample.sku}`)
    console.log(`   Title: ${sample.title}`)
    console.log(`   Has embedding: ${sample.embedding ? "Yes" : "No"}`)
    console.log(`   Embedding dimensions: ${sample.embedding?.length || 0}`)
  }
}

// Main migration function
async function migrate(): Promise<void> {
  const startTime = Date.now()
  console.log("🚀 Starting Shopify product migration...\n")

  // Configuration
  const BATCH_SIZE = 50 // Process 50 products at a time
  const BATCH_DELAY = 2000 // 2 seconds delay between batches
  const sqliteDbPath = path.join(__dirname, "..", "easymart.db")

  // Validate environment variables
  if (!process.env.MONGODB_ATLAS_URI) {
    throw new Error("MONGODB_ATLAS_URI environment variable is required")
  }
  if (!process.env.GOOGLE_API_KEY) {
    throw new Error("GOOGLE_API_KEY environment variable is required")
  }

  let mongoClient: MongoClient | null = null

  try {
    // Step 1: Extract products from SQLite
    console.log("📦 Step 1: Extracting products from SQLite...")
    const sqliteProducts = extractProductsFromSQLite(sqliteDbPath)

    if (sqliteProducts.length === 0) {
      console.log("⚠ No products found in SQLite database. Exiting.")
      return
    }

    // Step 2: Transform products to MongoDB schema
    console.log("\n🔄 Step 2: Transforming product schema...")
    const mongoProducts = sqliteProducts.map(transformProductToMongoSchema)
    console.log(`Transformed ${mongoProducts.length} products`)

    // Step 3: Connect to MongoDB
    console.log("\n🔌 Step 3: Connecting to MongoDB Atlas...")
    mongoClient = new MongoClient(process.env.MONGODB_ATLAS_URI, {
      serverApi: {
        version: '1' as any,
        strict: true,
        deprecationErrors: true,
      },
      family: 4, // Force IPv4
      tls: true,
      tlsAllowInvalidCertificates: false,
    })
    await mongoClient.connect()
    await mongoClient.db("admin").command({ ping: 1 })
    console.log("✓ Connected to MongoDB Atlas")

    // Step 4: Ensure vector search index exists
    console.log("\n🔍 Step 4: Checking vector search index...")
    await ensureVectorSearchIndex(mongoClient)

    // Step 5: Clear existing data (optional - comment out to append instead)
    console.log("\n🗑️  Step 5: Clearing existing products...")
    const db = mongoClient.db("inventory_database")
    const collection = db.collection("items")
    const deleteResult = await collection.deleteMany({})
    console.log(`Deleted ${deleteResult.deletedCount} existing documents`)

    // Step 6: Initialize embeddings model
    console.log("\n🧠 Step 6: Initializing AI embeddings model...")
    const embeddingsModel = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_API_KEY,
      modelName: "text-embedding-004",
    })
    console.log("✓ Embeddings model ready")

    // Step 7: Process products in batches
    console.log(`\n⚙️  Step 7: Processing ${mongoProducts.length} products in batches of ${BATCH_SIZE}...`)
    let totalSuccess = 0
    let totalFailed = 0

    for (let i = 0; i < mongoProducts.length; i += BATCH_SIZE) {
      const batch = mongoProducts.slice(i, i + BATCH_SIZE)
      const batchNum = Math.floor(i / BATCH_SIZE) + 1
      const totalBatches = Math.ceil(mongoProducts.length / BATCH_SIZE)

      console.log(`\n📦 Batch ${batchNum}/${totalBatches} (${batch.length} products)`)

      // Generate embeddings for batch
      const productsWithEmbeddings = await generateEmbeddingsForBatch(
        batch,
        embeddingsModel
      )

      // Insert batch into MongoDB
      const result = await insertProductsToMongo(mongoClient, productsWithEmbeddings)
      totalSuccess += result.success
      totalFailed += result.failed

      // Delay between batches to respect rate limits
      if (i + BATCH_SIZE < mongoProducts.length) {
        console.log(`⏳ Waiting ${BATCH_DELAY / 1000}s before next batch...`)
        await new Promise((resolve) => setTimeout(resolve, BATCH_DELAY))
      }
    }

    // Step 8: Verify migration
    console.log("\n✅ Step 8: Verifying migration...")
    await verifyMigration(mongoClient)

    // Final report
    const duration = ((Date.now() - startTime) / 1000).toFixed(2)
    console.log(`\n${"=".repeat(60)}`)
    console.log("🎉 MIGRATION COMPLETE!")
    console.log(`${"=".repeat(60)}`)
    console.log(`✓ Successfully migrated: ${totalSuccess} products`)
    console.log(`✗ Failed: ${totalFailed} products`)
    console.log(`⏱️  Total time: ${duration}s`)
    console.log(`${"=".repeat(60)}\n`)
  } catch (error) {
    console.error("\n❌ Migration failed:", error)
    throw error
  } finally {
    // Cleanup
    if (mongoClient) {
      await mongoClient.close()
      console.log("Closed MongoDB connection")
    }
  }
}

// Execute migration
migrate().catch((error) => {
  console.error("Fatal error:", error)
  process.exit(1)
})
