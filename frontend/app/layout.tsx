import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono',
  display: 'swap',
})

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || 'http://localhost:3000'),
  title: 'Nexsure | AI Insurance Risk Intelligence',
  description: 'Production-grade AI underwriting intelligence platform.',
  keywords: [
    'AI insurance',
    'health insurance risk assessment',
    'machine learning',
    'SHAP explainability',
    'Nexsure',
  ],
  authors: [{ name: 'Nexsure Engineering' }],
  robots: 'index, follow',
  openGraph: {
    title: 'Nexsure | AI Insurance Risk Intelligence',
    description: 'Production-grade AI underwriting intelligence platform.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={`${inter.variable} ${jetbrainsMono.variable}`}>
      <body className="antialiased grid-bg noise-overlay">
        {children}
      </body>
    </html>
  )
}
