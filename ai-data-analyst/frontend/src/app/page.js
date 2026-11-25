import Hero from '@/components/landing/Hero';
import AntigravityScroll from '@/components/landing/AntigravityScroll';
import Pricing from '@/components/landing/Pricing';
import Link from 'next/link';
import { REPO_URL } from '@/lib/site';
import styles from './beta-home.module.css';

const PRIVATE_BETA = process.env.NEXT_PUBLIC_PRIVATE_BETA !== 'false';

export default function Home() {
  if (PRIVATE_BETA) {
    return (
      <div className={styles.page}>
        <section className={styles.hero}>
          <p className={styles.eyebrow}>NEURALANALYST · OPEN SOURCE AI DATA ANALYST</p>
          <h1>Ask questions of your data. Check the evidence.</h1>
          <p className={styles.intro}>
            A language model turns your question into a validated plan; deterministic code computes
            every number; the answer shows the filters, rows and tables behind it. Train models with
            leakage checks and calibration, then score new files with a drift check.
          </p>
          <div className={styles.actions}>
            <Link className={styles.primary} href="/app/datasets">Open the app</Link>
            <a className={styles.secondary} href={REPO_URL} target="_blank" rel="noreferrer">View on GitHub</a>
          </div>
        </section>
        <section className={styles.details} aria-label="What Neural Analyst does">
          <div><span>01</span><h2>Ask</h2><p>Totals, rates, margins, trends and period comparisons in plain language, with the filters stated back.</p></div>
          <div><span>02</span><h2>Explain</h2><p>&ldquo;Why did sales drop?&rdquo; checks every dimension and leads with where the change is concentrated.</p></div>
          <div><span>03</span><h2>Predict</h2><p>Baseline-checked models with a tuned cut-off, then batch scoring of new files with drift and trust checks.</p></div>
        </section>
      </div>
    );
  }

  return (
    <>
      <Hero />
      <AntigravityScroll />
      <Pricing />
    </>
  );
}
