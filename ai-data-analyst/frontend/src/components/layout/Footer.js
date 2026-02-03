import Link from 'next/link';
import styles from './Footer.module.css';

const footerLinks = {
    product: [
        { href: '/features', label: 'Features' },
        { href: '/pricing', label: 'Pricing' },
        { href: '/integrations', label: 'Integrations' },
        { href: '/changelog', label: 'Changelog' },
    ],
    company: [
        { href: '/about', label: 'About' },
        { href: '/blog', label: 'Blog' },
        { href: '/careers', label: 'Careers' },
        { href: '/contact', label: 'Contact' },
    ],
    resources: [
        { href: '/docs', label: 'Documentation' },
        { href: '/help', label: 'Help Center' },
        { href: '/status', label: 'Status' },
        { href: '/security', label: 'Security' },
    ],
    legal: [
        { href: '/terms', label: 'Terms' },
        { href: '/privacy', label: 'Privacy' },
        { href: '/cookies', label: 'Cookies' },
    ],
};

export default function Footer() {
    return (
        <footer className={styles.footer}>
            <div className={styles.container}>
                <div className={styles.top}>
                    <div className={styles.brand}>
                        <Link href="/" className={styles.logo}>
                            <span className={styles.logoText}>Neural<span className={styles.logoAccent}>Analyst</span></span>
                        </Link>
                        <p className={styles.tagline}>
                            AI-powered data analysis that works while you sleep.
                        </p>
                    </div>

                    <div className={styles.links}>
                        <div className={styles.linkGroup}>
                            <h4 className={styles.linkTitle}>Product</h4>
                            <ul>
                                {footerLinks.product.map((link) => (
                                    <li key={link.href}>
                                        <Link href={link.href} className={styles.link}>{link.label}</Link>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className={styles.linkGroup}>
                            <h4 className={styles.linkTitle}>Company</h4>
                            <ul>
                                {footerLinks.company.map((link) => (
                                    <li key={link.href}>
                                        <Link href={link.href} className={styles.link}>{link.label}</Link>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className={styles.linkGroup}>
                            <h4 className={styles.linkTitle}>Resources</h4>
                            <ul>
                                {footerLinks.resources.map((link) => (
                                    <li key={link.href}>
                                        <Link href={link.href} className={styles.link}>{link.label}</Link>
                                    </li>
                                ))}
                            </ul>
                        </div>

                        <div className={styles.linkGroup}>
                            <h4 className={styles.linkTitle}>Legal</h4>
                            <ul>
                                {footerLinks.legal.map((link) => (
                                    <li key={link.href}>
                                        <Link href={link.href} className={styles.link}>{link.label}</Link>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                </div>

                <div className={styles.bottom}>
                    <p className={styles.copyright}>
                        © {new Date().getFullYear()} NeuralAnalyst. Crafted with care.
                    </p>
                </div>
            </div>
        </footer>
    );
}
