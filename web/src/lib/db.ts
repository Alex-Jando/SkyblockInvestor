import { Pool } from "pg";

const globalForPg = globalThis as unknown as { pool?: Pool };

export function getPool(): Pool {
  const connectionString = process.env.DATABASE_URL;
  if (!connectionString) {
    throw new Error("DATABASE_URL is required for web API routes.");
  }

  if (!globalForPg.pool) {
    const needsSsl =
      !connectionString.includes("localhost") &&
      !connectionString.includes("127.0.0.1");
    globalForPg.pool = new Pool({
      connectionString,
      ssl: needsSsl ? { rejectUnauthorized: false } : undefined
    });
  }

  return globalForPg.pool;
}
