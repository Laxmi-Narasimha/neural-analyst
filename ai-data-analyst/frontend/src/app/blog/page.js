import Link from 'next/link';
import { IconTime, IconUser, IconArrowRight } from '@/components/icons';
import styles from './page.module.css';

export const metadata = {
    title: 'Blog - Data Analytics Insights & Tutorials',
    description: 'Latest articles on AI data analysis, machine learning, statistics, and best practices for data teams.',
    openGraph: {
        title: 'Blog | NeuralAnalyst',
        description: 'Insights and tutorials for data professionals.',
    },
};

const featuredPost = {
    slug: 'future-of-ai-data-analysis',
    title: 'The Future of AI-Powered Data Analysis: 2025 and Beyond',
    excerpt: 'Explore how AI is transforming data analysis with automated insights, natural language queries, and self-service analytics for every team.',
    author: 'Sarah Chen',
    date: 'Dec 5, 2024',
    readTime: '8 min read',
    category: 'AI & ML',
    image: '/blog/future-ai.jpg',
};

const posts = [
    {
        slug: 'ab-testing-guide',
        title: 'The Complete Guide to A/B Testing',
        excerpt: 'Learn statistical significance, sample size calculations, and common pitfalls in experiment design.',
        author: 'Michael Torres',
        date: 'Dec 3, 2024',
        readTime: '12 min read',
        category: 'Statistics',
    },
    {
        slug: 'automl-explained',
        title: 'AutoML Explained: How Automated Machine Learning Works',
        excerpt: 'Understand how AutoML selects algorithms, tunes hyperparameters, and builds production-ready models.',
        author: 'Dr. Emily Watson',
        date: 'Nov 28, 2024',
        readTime: '10 min read',
        category: 'Machine Learning',
    },
    {
        slug: 'customer-churn-prediction',
        title: 'Building a Customer Churn Prediction Model',
        excerpt: 'Step-by-step tutorial on predicting customer churn using NeuralAnalyst and machine learning.',
        author: 'Alex Kim',
        date: 'Nov 22, 2024',
        readTime: '15 min read',
        category: 'Tutorial',
    },
    {
        slug: 'data-cleaning-best-practices',
        title: '10 Data Cleaning Best Practices for Analysts',
        excerpt: 'Essential techniques for handling missing values, outliers, and data quality issues.',
        author: 'Sarah Chen',
        date: 'Nov 18, 2024',
        readTime: '7 min read',
        category: 'Best Practices',
    },
    {
        slug: 'time-series-forecasting',
        title: 'Time Series Forecasting: Prophet vs ARIMA',
        excerpt: 'Compare two popular forecasting methods and learn when to use each approach.',
        author: 'Dr. Emily Watson',
        date: 'Nov 12, 2024',
        readTime: '11 min read',
        category: 'Statistics',
    },
    {
        slug: 'byok-privacy-benefits',
        title: 'BYOK: Complete Privacy Control for Enterprise Data',
        excerpt: 'How Bring Your Own Key ensures your sensitive data never leaves your control.',
        author: 'Michael Torres',
        date: 'Nov 5, 2024',
        readTime: '6 min read',
        category: 'Security',
    },
];

const categories = [
    'All',
    'AI & ML',
    'Statistics',
    'Machine Learning',
    'Tutorial',
    'Best Practices',
    'Security',
];

export default function BlogPage() {
    return (
        <main className={styles.main}>
            {/* Hero */}
            <section className={styles.hero}>
                <div className={styles.container}>
                    <h1 className={styles.heroTitle}>
                        <span className={styles.gradient}>Blog</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        Insights, tutorials, and best practices for data professionals
                    </p>
                </div>
            </section>

            {/* Featured Post */}
            <section className={styles.featured}>
                <div className={styles.container}>
                    <Link href={`/blog/${featuredPost.slug}`} className={styles.featuredCard}>
                        <div className={styles.featuredImage}>
                            <span className={styles.featuredBadge}>Featured</span>
                        </div>
                        <div className={styles.featuredContent}>
                            <span className={styles.category}>{featuredPost.category}</span>
                            <h2 className={styles.featuredTitle}>{featuredPost.title}</h2>
                            <p className={styles.featuredExcerpt}>{featuredPost.excerpt}</p>
                            <div className={styles.meta}>
                                <span><IconUser size={14} /> {featuredPost.author}</span>
                                <span><IconTime size={14} /> {featuredPost.readTime}</span>
                            </div>
                        </div>
                    </Link>
                </div>
            </section>

            {/* Categories */}
            <section className={styles.categoriesSection}>
                <div className={styles.container}>
                    <div className={styles.categories}>
                        {categories.map((cat) => (
                            <button key={cat} className={styles.categoryBtn}>
                                {cat}
                            </button>
                        ))}
                    </div>
                </div>
            </section>

            {/* Posts Grid */}
            <section className={styles.posts}>
                <div className={styles.container}>
                    <div className={styles.postsGrid}>
                        {posts.map((post) => (
                            <Link key={post.slug} href={`/blog/${post.slug}`} className={styles.postCard}>
                                <span className={styles.postCategory}>{post.category}</span>
                                <h3 className={styles.postTitle}>{post.title}</h3>
                                <p className={styles.postExcerpt}>{post.excerpt}</p>
                                <div className={styles.postMeta}>
                                    <span>{post.author}</span>
                                    <span>{post.date}</span>
                                </div>
                                <span className={styles.postRead}>
                                    Read article <IconArrowRight size={14} />
                                </span>
                            </Link>
                        ))}
                    </div>
                </div>
            </section>

            {/* Newsletter */}
            <section className={styles.newsletter}>
                <div className={styles.container}>
                    <div className={styles.newsletterCard}>
                        <h2>Subscribe to our newsletter</h2>
                        <p>Get the latest articles and updates delivered to your inbox.</p>
                        <form className={styles.newsletterForm}>
                            <input type="email" placeholder="your@email.com" />
                            <button type="submit">Subscribe</button>
                        </form>
                    </div>
                </div>
            </section>
        </main>
    );
}
