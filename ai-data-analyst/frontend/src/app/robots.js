export default function robots() {
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
        sitemap: 'https://neuralanalyst.ai/sitemap.xml',
        host: 'https://neuralanalyst.ai',
    };
}
