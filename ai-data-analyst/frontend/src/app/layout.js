import '@/styles/globals.css';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import FluidBackground from '@/components/effects/FluidBackground';

export const metadata = {
  metadataBase: new URL('https://neuralanalyst.ai'),
  title: {
    default: 'NeuralAnalyst - Your 24/7 AI Data Analyst',
    template: '%s | NeuralAnalyst',
  },
  description: 'NeuralAnalyst is an AI-powered data analyst that works 24/7. Analyze millions of data points, generate insights, and build ML models without coding. Free tier available. BYOK supported.',
  keywords: [
    'AI data analyst',
    'machine learning',
    'data analysis',
    'automated analytics',
    'business intelligence',
    'data visualization',
    'predictive analytics',
    'no-code AI',
    'BYOK',
    'OpenAI',
    'Gemini',
    'data science',
    'AutoML',
  ],
  authors: [{ name: 'NeuralAnalyst Team' }],
  creator: 'NeuralAnalyst',
  publisher: 'NeuralAnalyst',
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://neuralanalyst.ai',
    siteName: 'NeuralAnalyst',
    title: 'NeuralAnalyst - Your 24/7 AI Data Analyst',
    description: 'AI-powered data analyst that works around the clock. 244+ features, 3.5M rows/hour, enterprise-grade security.',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'NeuralAnalyst - AI Data Analyst',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'NeuralAnalyst - Your 24/7 AI Data Analyst',
    description: 'AI-powered data analyst that works around the clock. 244+ features, no coding required.',
    images: ['/twitter-image.png'],
    creator: '@neuralanalyst',
  },
  verification: {
    google: 'google-site-verification-code',
  },
  alternates: {
    canonical: 'https://neuralanalyst.ai',
  },
  category: 'technology',
};

export const viewport = {
  themeColor: '#030712',
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
};

// JSON-LD Structured Data
const jsonLd = {
  '@context': 'https://schema.org',
  '@type': 'SoftwareApplication',
  name: 'NeuralAnalyst',
  applicationCategory: 'BusinessApplication',
  operatingSystem: 'Web',
  description: 'AI-powered data analyst with 244+ features for automated data analysis, machine learning, and visualization.',
  offers: {
    '@type': 'Offer',
    price: '0',
    priceCurrency: 'USD',
    description: 'Free tier available. Pro starts at $29/month.',
  },
  aggregateRating: {
    '@type': 'AggregateRating',
    ratingValue: '4.9',
    ratingCount: '1250',
    bestRating: '5',
    worstRating: '1',
  },
  featureList: [
    '244+ Data Analysis Features',
    '24/7 Automated Analysis',
    'Natural Language Queries',
    'Machine Learning & AutoML',
    'Real-time Dashboards',
    'BYOK - Use Your Own API Keys',
  ],
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" sizes="any" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
        <script
          type="application/ld+json"
          dangerouslySetInnerHTML={{ __html: JSON.stringify(jsonLd) }}
        />
      </head>
      <body suppressHydrationWarning>
        <FluidBackground />
        <Header />
        <main style={{ position: 'relative', zIndex: 1 }}>{children}</main>
        <Footer />
      </body>
    </html>
  );
}

