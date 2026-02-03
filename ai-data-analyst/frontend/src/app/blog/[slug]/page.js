import Link from 'next/link';
import { IconArrowRight, IconTime, IconUser } from '@/components/icons';
import styles from './page.module.css';

// Blog post content (in a real app, this would come from a CMS)
const posts = {
    'future-of-ai-data-analysis': {
        title: 'The Future of AI-Powered Data Analysis: 2025 and Beyond',
        author: 'Sarah Chen',
        date: 'December 5, 2024',
        readTime: '8 min read',
        category: 'AI & ML',
        content: `
The landscape of data analysis is undergoing a fundamental transformation. As we approach 2025, AI-powered tools are making sophisticated analysis accessible to teams without dedicated data scientists.

## The Democratization of Data

Traditionally, extracting insights from data required specialized skills in statistics, programming, and domain expertise. Today, natural language interfaces are bridging this gap, allowing anyone to ask questions like "What's driving our customer churn?" and receive actionable answers.

## Key Trends Shaping the Future

### 1. Natural Language Queries
The ability to analyze data using plain English is becoming standard. Rather than writing SQL or Python, analysts can simply describe what they want to know.

### 2. Automated Insight Generation
AI systems now proactively identify patterns, anomalies, and opportunities in your data—surfacing insights you didn't know to look for.

### 3. Self-Service ML
Building machine learning models no longer requires a PhD. AutoML platforms handle algorithm selection, feature engineering, and hyperparameter tuning automatically.

### 4. Real-Time Analysis
The shift from batch processing to real-time analysis means businesses can respond to changes as they happen, not days later.

## What This Means for Your Team

The implications are significant. Data teams will shift from writing queries to validating insights. Business users will become more data-independent. And the speed of decision-making will accelerate dramatically.

## Getting Started

The best way to prepare is to start experimenting today. Tools like NeuralAnalyst offer free tiers that let you experience the future of data analysis firsthand.

The question isn't whether AI will transform data analysis—it's whether your organization will lead or follow.
    `,
    },
    'ab-testing-guide': {
        title: 'The Complete Guide to A/B Testing',
        author: 'Michael Torres',
        date: 'December 3, 2024',
        readTime: '12 min read',
        category: 'Statistics',
        content: `
A/B testing is the gold standard for making data-driven decisions. This comprehensive guide covers everything from experiment design to statistical analysis.

## What is A/B Testing?

A/B testing (also called split testing) compares two versions of something to determine which performs better. Version A is typically the control, while version B includes the change you want to test.

## When to Use A/B Testing

A/B tests are ideal when you:
- Have a specific hypothesis to test
- Can measure a clear outcome metric
- Have enough traffic/data for statistical significance
- Can randomize users into groups

## Designing Your Experiment

### 1. Define Your Hypothesis
Start with a clear, testable statement: "Adding a progress bar will increase form completion rate."

### 2. Choose Your Metrics
Primary metric: The main outcome you're measuring
Secondary metrics: Other outcomes to monitor
Guardrail metrics: Metrics that shouldn't decrease

### 3. Calculate Sample Size
Use power analysis to determine how many observations you need. The required sample size depends on:
- Expected effect size
- Baseline conversion rate
- Desired statistical power (typically 80%)
- Significance level (typically 5%)

## Avoiding Common Pitfalls

### Peeking Problem
Looking at results before the test is complete inflates false positive rates. Define your sample size upfront and wait.

### Multiple Comparisons
Testing many variations increases the chance of false positives. Use correction methods like Bonferroni or sequential testing.

### Selection Bias
Ensure randomization is truly random. Watch for technical issues that might bias group assignment.

## Analyzing Results

Calculate the difference between groups and determine statistical significance. Remember: statistical significance doesn't always mean practical significance.

## Conclusion

A/B testing is a powerful tool when used correctly. Start with clear hypotheses, design rigorous experiments, and let the data guide your decisions.
    `,
    },
    'automl-explained': {
        title: 'AutoML Explained: How Automated Machine Learning Works',
        author: 'Dr. Emily Watson',
        date: 'November 28, 2024',
        readTime: '10 min read',
        category: 'Machine Learning',
        content: `
AutoML (Automated Machine Learning) is revolutionizing how organizations build predictive models. This article explains how it works under the hood.

## The AutoML Pipeline

AutoML automates the entire machine learning workflow:

### 1. Data Preprocessing
- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Feature engineering

### 2. Algorithm Selection
AutoML tries multiple algorithms and selects the best performer:
- Linear models (Logistic Regression, Linear Regression)
- Tree-based models (Random Forest, XGBoost, LightGBM)
- Neural networks (for complex patterns)
- Support Vector Machines

### 3. Hyperparameter Tuning
Each algorithm has settings that affect performance. AutoML optimizes these using:
- Grid search
- Random search
- Bayesian optimization
- Evolutionary algorithms

### 4. Model Validation
Rigorous cross-validation ensures the model generalizes well:
- K-fold cross-validation
- Hold-out validation
- Time-series aware splitting

## Benefits of AutoML

### Speed
What took data scientists weeks now takes hours.

### Consistency
Automated processes are reproducible and less prone to human error.

### Accessibility
Teams without ML expertise can build production-ready models.

## When AutoML Works Best

AutoML excels at tabular data problems: classification, regression, and time series forecasting. It handles the 80% of ML problems that follow common patterns.

## Limitations

AutoML isn't magic. It works best when:
- You have clean, structured data
- The problem is well-defined
- You can validate results with domain expertise

## Getting Started

NeuralAnalyst includes AutoML capabilities that let you build models with natural language. Simply describe your prediction goal, and the system handles the rest.
    `,
    },
};

export async function generateStaticParams() {
    return Object.keys(posts).map((slug) => ({ slug }));
}

export async function generateMetadata({ params }) {
    const post = posts[params.slug];
    if (!post) {
        return { title: 'Post Not Found' };
    }
    return {
        title: `${post.title} - NeuralAnalyst Blog`,
        description: post.content.substring(0, 160) + '...',
    };
}

export default function BlogPostPage({ params }) {
    const post = posts[params.slug];

    if (!post) {
        return (
            <main className={styles.main}>
                <div className={styles.container}>
                    <h1>Post not found</h1>
                    <Link href="/blog">Back to blog</Link>
                </div>
            </main>
        );
    }

    return (
        <main className={styles.main}>
            <article className={styles.article}>
                <div className={styles.container}>
                    {/* Header */}
                    <header className={styles.header}>
                        <Link href="/blog" className={styles.backLink}>
                            <IconArrowRight size={14} className={styles.backIcon} />
                            Back to Blog
                        </Link>
                        <span className={styles.category}>{post.category}</span>
                        <h1 className={styles.title}>{post.title}</h1>
                        <div className={styles.meta}>
                            <span><IconUser size={16} /> {post.author}</span>
                            <span>{post.date}</span>
                            <span><IconTime size={16} /> {post.readTime}</span>
                        </div>
                    </header>

                    {/* Content */}
                    <div className={styles.content}>
                        {post.content.split('\n').map((paragraph, i) => {
                            if (paragraph.startsWith('## ')) {
                                return <h2 key={i}>{paragraph.replace('## ', '')}</h2>;
                            }
                            if (paragraph.startsWith('### ')) {
                                return <h3 key={i}>{paragraph.replace('### ', '')}</h3>;
                            }
                            if (paragraph.startsWith('- ')) {
                                return <li key={i}>{paragraph.replace('- ', '')}</li>;
                            }
                            if (paragraph.trim()) {
                                return <p key={i}>{paragraph}</p>;
                            }
                            return null;
                        })}
                    </div>

                    {/* Share */}
                    <div className={styles.share}>
                        <span>Share this article:</span>
                        <div className={styles.shareButtons}>
                            <button>Twitter</button>
                            <button>LinkedIn</button>
                            <button>Copy Link</button>
                        </div>
                    </div>
                </div>
            </article>
        </main>
    );
}
