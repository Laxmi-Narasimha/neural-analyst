'use client';

import { usePathname } from 'next/navigation';

// Marketing chrome (animated background, site header, footer) belongs to the public
// pages only. The signed-in app under /app has its own shell in app/app/layout.js.
function isAppRoute(pathname) {
    return pathname === '/app' || Boolean(pathname?.startsWith('/app/'));
}

export default function SiteChrome({ background, header, footer, children }) {
    const pathname = usePathname();

    if (isAppRoute(pathname)) {
        return children;
    }

    return (
        <>
            {background}
            {header}
            <main style={{ position: 'relative', zIndex: 1 }}>{children}</main>
            {footer}
        </>
    );
}
