import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Nexsure | AI Insurance Risk Intelligence',
  description:
    'Production-grade autonomous AI platform for insurance risk assessment, explainability, and observability.',
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
    description: 'Production-grade autonomous AI platform for insurance risk assessment, explainability, and observability.',
    type: 'website',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable}>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link
          href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased grid-bg noise-overlay">
        {children}
      </body>
    </html>
  )
}