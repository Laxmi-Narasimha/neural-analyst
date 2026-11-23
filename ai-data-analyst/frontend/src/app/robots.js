export default function robots() {
    if (process.env.NEXT_PUBLIC_PRIVATE_BETA !== 'false') {
        return { rules: [{ userAgent: '*', disallow: '/' }] };
    }

    const baseUrl = process.env.NEXT_PUBLIC_SITE_URL
        || (process.env.VERCEL_URL ? `https://${process.env.VERCEL_URL}` : 'http://localhost:3000');
    return {
        rules: [
            {
                userAgent: '*',
                allow: '/',
                disallow: ['/app/', '/api/', '/admin/'],
            },
            {
                userAgent: 'Googlebot',
                allow: '/',
                disallow: ['/app/', '/api/', '/admin/'],
            },
        ],
        sitemap: `${baseUrl}/sitemap.xml`,
        host: baseUrl,
    };
}
