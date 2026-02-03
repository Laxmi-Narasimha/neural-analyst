'use client';

import { motion } from 'framer-motion';
import styles from './Comparison.module.css';
import { IconTime, IconLightning, IconFlask, IconMoney, IconTarget, IconTrend, IconUser, IconRobot } from '@/components/icons';

const comparisonData = [
    {
        aspect: 'Availability',
        human: '8 hours/day',
        humanDetail: 'Needs breaks, weekends, vacations',
        ai: '24/7/365',
        aiDetail: 'Works while you sleep, never takes a day off',
        Icon: IconTime,
    },
    {
        aspect: 'Processing Speed',
        human: '~100 rows/min',
        humanDetail: 'Manual analysis, prone to fatigue',
        ai: '3.5M rows/hour',
        aiDetail: 'Handles massive datasets instantly',
        Icon: IconLightning,
    },
    {
        aspect: 'Analysis Depth',
        human: '10-20 techniques',
        humanDetail: 'Limited by individual expertise',
        ai: '244+ features',
        aiDetail: 'ML, statistics, forecasting, NLP & more',
        Icon: IconFlask,
    },
    {
        aspect: 'Annual Cost',
        human: '$80,000+',
        humanDetail: 'Salary, benefits, training, turnover',
        ai: '$348/year',
        aiDetail: 'Pro plan at $29/month, free tier available',
        Icon: IconMoney,
    },
    {
        aspect: 'Fatigue & Errors',
        human: 'Increasing over time',
        humanDetail: 'Cognitive decline, inconsistent quality',
        ai: 'Consistent 99.9%',
        aiDetail: 'Same precision at midnight as at 9 AM',
        Icon: IconTarget,
    },
    {
        aspect: 'Scalability',
        human: 'Limited',
        humanDetail: 'Hiring takes months, training required',
        ai: 'Instant',
        aiDetail: 'Handle 10x workload with one click',
        Icon: IconTrend,
    },
];

export default function Comparison() {
    return (
        <section className={styles.section}>
            <div className={styles.container}>
                {/* Header */}
                <motion.div
                    className={styles.header}
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5 }}
                >
                    <span className={styles.label}>The Advantage</span>
                    <h2 className={styles.title}>
                        Human Analyst vs <span className={styles.gradient}>NeuralAnalyst</span>
                    </h2>
                    <p className={styles.subtitle}>
                        See why leading teams are augmenting their data capabilities with AI
                    </p>
                </motion.div>

                {/* Comparison Table */}
                <div className={styles.table}>
                    {/* Table Header */}
                    <div className={styles.tableHeader}>
                        <div className={styles.aspectCol}>Comparison</div>
                        <div className={styles.humanCol}>
                            <IconUser size={18} className={styles.headerIcon} />
                            Human Analyst
                        </div>
                        <div className={styles.aiCol}>
                            <IconRobot size={18} className={styles.headerIcon} />
                            NeuralAnalyst
                        </div>
                    </div>

                    {/* Table Rows */}
                    {comparisonData.map((row, index) => (
                        <motion.div
                            key={row.aspect}
                            className={styles.tableRow}
                            initial={{ opacity: 0, x: -20 }}
                            whileInView={{ opacity: 1, x: 0 }}
                            viewport={{ once: true }}
                            transition={{ duration: 0.4, delay: index * 0.1 }}
                        >
                            <div className={styles.aspectCol}>
                                <row.Icon size={18} className={styles.aspectIcon} />
                                <span className={styles.aspectName}>{row.aspect}</span>
                            </div>
                            <div className={styles.humanCol}>
                                <span className={styles.humanValue}>{row.human}</span>
                                <span className={styles.humanDetail}>{row.humanDetail}</span>
                            </div>
                            <div className={styles.aiCol}>
                                <span className={styles.aiValue}>{row.ai}</span>
                                <span className={styles.aiDetail}>{row.aiDetail}</span>
                            </div>
                        </motion.div>
                    ))}
                </div>

                {/* Bottom CTA */}
                <motion.div
                    className={styles.cta}
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.5, delay: 0.3 }}
                >
                    <p className={styles.ctaText}>
                        Don't replace your analysts — <strong>supercharge</strong> them with AI
                    </p>
                </motion.div>
            </div>
        </section>
    );
}
