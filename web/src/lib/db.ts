import { Pool } from "pg";

const globalForPg = globalThis as unknown as { pool?: Pool };

function normalizeConnectionString(raw: string): string {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    throw new Error("DATABASE_URL is not a valid Postgres URI.");
  }

  if (!url.protocol.startsWith("postgres")) {
    throw new Error("DATABASE_URL must use postgres:// or postgresql://");
  }

  // Avoid pg/pg-connection-string SSL mode ambiguity in serverless environments.
  // We control TLS behavior explicitly via the Pool `ssl` option below.
  url.searchParams.delete("sslmode");
  url.searchParams.delete("uselibpqcompat");
  url.searchParams.delete("sslcert");
  url.searchParams.delete("sslkey");
  url.searchParams.delete("sslrootcert");

  return url.toString();
}

export function getPool(): Pool {
  const rawConnectionString = process.env.DATABASE_URL;
  if (!rawConnectionString) {
    throw new Error("DATABASE_URL is required for web API routes.");
  }
  const connectionString = normalizeConnectionString(rawConnectionString);

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
