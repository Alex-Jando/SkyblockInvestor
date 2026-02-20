import type { Metadata } from "next";
import Link from "next/link";
import { IBM_Plex_Serif, Space_Grotesk } from "next/font/google";

import "./globals.css";

const sans = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans"
});

const serif = IBM_Plex_Serif({
  subsets: ["latin"],
  variable: "--font-serif",
  weight: ["400", "600"]
});

export const metadata: Metadata = {
  title: "SkyBlock Bazaar Investment Basket",
  description: "Daily long-horizon Hypixel SkyBlock Bazaar basket signals."
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body
        style={{
          fontFamily: "var(--font-sans)"
        }}
      >
        <div className="app-shell">
          <div className="app-inner">
            <nav className="top-nav">
              <div
                className="brand"
                style={{ fontFamily: "var(--font-serif)" }}
              >
                SkyBlock Investor
              </div>
              <div className="nav-links">
                <Link className="nav-link" href="/">
                  Basket
                </Link>
                <Link className="nav-link" href="/sell">
                  Sell Signals
                </Link>
                <Link className="nav-link" href="/performance">
                  Performance
                </Link>
                <Link className="nav-link" href="/how">
                  How It Works
                </Link>
              </div>
            </nav>
            {children}
          </div>
        </div>
      </body>
    </html>
  );
}
