'use client';

import { useState } from 'react';
import { IconArrowRight, IconPlus, IconMinus } from '@/components/icons';
import styles from './page.module.css';

const faqCategories = [
    {
        name: 'General',
        questions: [
            { q: 'What is NeuralAnalyst?', a: 'NeuralAnalyst is an AI-powered data analysis platform that lets you analyze data, build machine learning models, and generate insights using natural language queries.' },
            { q: 'Do I need coding experience?', a: 'No! NeuralAnalyst is designed for everyone. You can analyze data using plain English questions without writing any code.' },
            { q: 'What file formats are supported?', a: 'We support CSV, Excel (.xlsx, .xls), JSON, Parquet, and can connect directly to databases like PostgreSQL, MySQL, and Snowflake.' },
            { q: 'Is there a free plan?', a: 'Yes! Our free plan includes 3 analyses per month, up to 10MB datasets, and basic features. Perfect for getting started.' },
        ],
    },
    {
        name: 'BYOK & Privacy',
        questions: [
            { q: 'What is BYOK mode?', a: 'BYOK (Bring Your Own Key) mode lets you use your own API keys from OpenAI, Google, or Anthropic. Your data never touches our servers.' },
            { q: 'How does BYOK protect my data?', a: 'In BYOK mode, all AI processing happens through your own API accounts. We never see, store, or process your data.' },
            { q: 'Which AI providers are supported?', a: 'We support OpenAI (GPT-4, GPT-3.5), Google (Gemini Pro, Gemini Ultra), and Anthropic (Claude 3).' },
            { q: 'Where are API keys stored?', a: 'API keys are stored locally in your browser and are never sent to our servers.' },
        ],
    },
    {
        name: 'Pricing & Billing',
        questions: [
            { q: 'Can I upgrade or downgrade anytime?', a: 'Yes, you can change your plan at any time. Upgrades are effective immediately, downgrades take effect at the next billing cycle.' },
            { q: 'What payment methods are accepted?', a: 'We accept all major credit cards (Visa, MasterCard, Amex) through Stripe. Enterprise customers can pay by invoice.' },
            { q: 'Is there a refund policy?', a: 'We offer a 14-day money-back guarantee for all paid plans. Contact support for refund requests.' },
            { q: 'Do you offer discounts for nonprofits?', a: 'Yes! We offer 50% off for verified nonprofits and educational institutions.' },
        ],
    },
    {
        name: 'Security',
        questions: [
            { q: 'Is my data secure?', a: 'Yes. We use 256-bit AES encryption at rest and TLS 1.3 in transit. We are also pursuing SOC 2 Type II certification.' },
            { q: 'Do you share data with third parties?', a: 'We never sell your data. Data may be shared with AI providers (for managed plans) solely to provide the service.' },
            { q: 'Can I delete my data?', a: 'Yes, you can delete your data anytime from Settings. All data is removed within 30 days of deletion.' },
            { q: 'Do you have SSO?', a: 'Yes, SSO via SAML and OIDC is available on Enterprise plans.' },
        ],
    },
];

export default function FAQPage() {
    const [openItems, setOpenItems] = useState({});

    const toggleItem = (categoryIdx, questionIdx) => {
        const key = `${categoryIdx}-${questionIdx}`;
        setOpenItems(prev => ({ ...prev, [key]: !prev[key] }));
    };

    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        Frequently Asked <span className={styles.gradient}>Questions</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Find answers to common questions about NeuralAnalyst
                    </p>
                </div>
            </section>

            {/* FAQ Sections */}
            <section className={styles.faq}>
                <div className={styles.container}>
                    {faqCategories.map((category, catIdx) => (
                        <div key={catIdx} className={styles.category}>
                            <h2 className={styles.categoryTitle}>{category.name}</h2>
                            <div className={styles.questionsList}>
                                {category.questions.map((item, qIdx) => {
                                    const isOpen = openItems[`${catIdx}-${qIdx}`];
                                    return (
                                        <div key={qIdx} className={styles.questionItem}>
                                            <button
                                                className={styles.questionBtn}
                                                onClick={() => toggleItem(catIdx, qIdx)}
                                                aria-expanded={isOpen}
                                            >
                                                <span>{item.q}</span>
                                                {isOpen ? <IconMinus size={18} /> : <IconPlus size={18} />}
                                            </button>
                                            {isOpen && (
                                                <div className={styles.answer}>
                                                    <p>{item.a}</p>
                                                </div>
                                            )}
                                        </div>
                                    );
                                })}
                            </div>
                        </div>
                    ))}
                </div>
            </section>

            {/* Contact */}
            <section className={styles.contact}>
                <div className={styles.container}>
                    <h2>Still have questions?</h2>
                    <p>Contact our support team for personalized help.</p>
                    <a href="/contact" className={styles.contactBtn}>
                        Contact Support
                        <IconArrowRight size={18} />
                    </a>
                </div>
            </section>
        </main>
    );
}
