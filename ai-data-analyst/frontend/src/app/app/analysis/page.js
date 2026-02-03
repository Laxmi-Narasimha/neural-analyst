'use client';

import { useState } from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';
import { IconPlus, IconSearch, IconChart, IconArrowRight, IconTime } from '@/components/icons';
import styles from './page.module.css';

const analyses = [
    { id: 1, name: 'Q4 Sales Analysis', type: 'Exploratory Data Analysis', dataset: 'sales_2024.csv', date: '2 hours ago', status: 'complete' },
    { id: 2, name: 'Customer Churn Model', type: 'Machine Learning', dataset: 'customers.parquet', date: '1 day ago', status: 'complete' },
    { id: 3, name: 'Marketing A/B Test', type: 'Statistical Testing', dataset: 'campaigns.csv', date: '3 days ago', status: 'complete' },
    { id: 4, name: 'Revenue Forecast', type: 'Time Series', dataset: 'monthly_revenue.csv', date: '5 days ago', status: 'complete' },
    { id: 5, name: 'Sentiment Analysis', type: 'NLP', dataset: 'customer_reviews.json', date: '1 week ago', status: 'complete' },
];

const analysisTypes = [
    { id: 'all', label: 'All' },
    { id: 'eda', label: 'EDA' },
    { id: 'ml', label: 'Machine Learning' },
    { id: 'stats', label: 'Statistics' },
    { id: 'timeseries', label: 'Time Series' },
    { id: 'nlp', label: 'NLP' },
];

export default function AnalysisPage() {
    const [activeType, setActiveType] = useState('all');
    const [searchQuery, setSearchQuery] = useState('');

    return (
        <div className={styles.page}>
            {/* Header */}
            <div className={styles.header}>
                <div>
                    <h1 className={styles.title}>Analysis History</h1>
                    <p className={styles.subtitle}>View and manage your data analyses</p>
                </div>
                <Link href="/app/analysis/new" className={styles.newBtn}>
                    <IconPlus size={18} />
                    New Analysis
                </Link>
            </div>

            {/* Filters */}
            <div className={styles.filters}>
                <div className={styles.tabs}>
                    {analysisTypes.map((type) => (
                        <button
                            key={type.id}
                            className={`${styles.tab} ${activeType === type.id ? styles.active : ''}`}
                            onClick={() => setActiveType(type.id)}
                        >
                            {type.label}
                        </button>
                    ))}
                </div>
                <div className={styles.searchWrapper}>
                    <IconSearch size={18} className={styles.searchIcon} />
                    <input
                        type="text"
                        placeholder="Search..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        className={styles.searchInput}
                    />
                </div>
            </div>

            {/* Analysis List */}
            <div className={styles.analysisList}>
                {analyses.map((analysis, index) => (
                    <motion.div
                        key={analysis.id}
                        className={styles.analysisCard}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.05 }}
                    >
                        <div className={styles.analysisIcon}>
                            <IconChart size={20} />
                        </div>
                        <div className={styles.analysisInfo}>
                            <h3 className={styles.analysisName}>{analysis.name}</h3>
                            <p className={styles.analysisMeta}>
                                {analysis.type} • {analysis.dataset}
                            </p>
                        </div>
                        <div className={styles.analysisTime}>
                            <IconTime size={14} />
                            <span>{analysis.date}</span>
                        </div>
                        <span className={`${styles.status} ${styles[analysis.status]}`}>
                            {analysis.status}
                        </span>
                        <Link href={`/app/analysis/${analysis.id}`} className={styles.analysisAction}>
                            <IconArrowRight size={18} />
                        </Link>
                    </motion.div>
                ))}
            </div>
        </div>
    );
}
