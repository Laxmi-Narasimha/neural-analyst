import '@/styles/globals.css';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import FluidBackground from '@/components/effects/FluidBackground';

const PRIVATE_BETA = process.env.NEXT_PUBLIC_PRIVATE_BETA !== 'false';
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL
  || (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 'http://localhost:3000');

export const metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'NeuralAnalyst - Private Beta',
    template: '%s | NeuralAnalyst',
  },
  description: 'NeuralAnalyst helps invited beta users explore datasets and review evidence behind analytical results.',
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
    index: !PRIVATE_BETA,
    follow: !PRIVATE_BETA,
    googleBot: {
      index: !PRIVATE_BETA,
      follow: !PRIVATE_BETA,
      'max-video-preview': -1,
      'max-image-preview': 'large',
      'max-snippet': -1,
    },
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: SITE_URL,
    siteName: 'NeuralAnalyst',
    title: 'NeuralAnalyst - Private Beta',
    description: 'Explore datasets with traceable analytical results during our private beta.',
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
    title: 'NeuralAnalyst - Private Beta',
    description: 'Explore datasets with traceable analytical results during our private beta.',
    images: ['/twitter-image.png'],
  },
  alternates: {
    canonical: SITE_URL,
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
  description: 'Private beta for dataset exploration and evidence-backed analysis.',
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

