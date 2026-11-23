import Hero from '@/components/landing/Hero';
import AntigravityScroll from '@/components/landing/AntigravityScroll';
import Pricing from '@/components/landing/Pricing';
import Link from 'next/link';
import styles from './beta-home.module.css';

const PRIVATE_BETA = process.env.NEXT_PUBLIC_PRIVATE_BETA !== 'false';

export default function Home() {
  if (PRIVATE_BETA) {
    return (
      <div className={styles.page}>
        <section className={styles.hero}>
          <p className={styles.eyebrow}>NEURALANALYST · PRIVATE BETA</p>
          <h1>Ask questions of your data. Check the evidence.</h1>
          <p className={styles.intro}>
            Upload a dataset, explore it in plain language, and inspect the calculations
            behind the answer. We are testing this workflow with a small group before
            opening public access.
          </p>
          <div className={styles.actions}>
            <Link className={styles.primary} href="/register">Use an invite code</Link>
            <Link className={styles.secondary} href="/login">Sign in</Link>
          </div>
        </section>
        <section className={styles.details} aria-label="What the beta includes">
          <div><span>01</span><h2>Explore</h2><p>Start with CSV data and ask natural language questions.</p></div>
          <div><span>02</span><h2>Verify</h2><p>Review the result and the analytical steps used to produce it.</p></div>
          <div><span>03</span><h2>Improve</h2><p>Tell us where the workflow saves time and where it falls short.</p></div>
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
