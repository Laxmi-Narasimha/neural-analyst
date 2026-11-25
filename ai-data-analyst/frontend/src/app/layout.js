import '@/styles/globals.css';
import Header from '@/components/layout/Header';
import Footer from '@/components/layout/Footer';
import FluidBackground from '@/components/effects/FluidBackground';
import SiteChrome from '@/components/layout/SiteChrome';

const PRIVATE_BETA = process.env.NEXT_PUBLIC_PRIVATE_BETA !== 'false';
const SITE_URL = process.env.NEXT_PUBLIC_SITE_URL
  || (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 'http://localhost:3000');

export const metadata = {
  metadataBase: new URL(SITE_URL),
  title: {
    default: 'NeuralAnalyst - Open-source AI data analyst',
    template: '%s | NeuralAnalyst',
  },
  description: 'Open-source AI data analyst: plain-language questions answered by validated plans and deterministic code, with the evidence shown.',
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
    title: 'NeuralAnalyst - Open-source AI data analyst',
    description: 'Ask questions of your data and check the evidence behind every answer.',
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
    title: 'NeuralAnalyst - Open-source AI data analyst',
    description: 'Ask questions of your data and check the evidence behind every answer.',
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
  description: 'Open-source AI data analyst with evidence-backed answers.',
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
        <SiteChrome background={<FluidBackground />} header={<Header />} footer={<Footer />}>
          {children}
        </SiteChrome>
      </body>
    </html>
  );
}

