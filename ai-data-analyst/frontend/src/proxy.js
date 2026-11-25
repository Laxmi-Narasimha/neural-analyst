import { NextResponse } from 'next/server';

// Legacy marketing pages include claims and plan promises that are not approved
// for the private beta. Keep them unavailable until they have been reviewed.
export function proxy(request) {
  if (process.env.NEXT_PUBLIC_PRIVATE_BETA !== 'false') {
    return NextResponse.redirect(new URL('/', request.url));
  }
  return NextResponse.next();
}

export const config = {
  matcher: [
    '/about', '/features', '/solutions/:path*', '/integrations', '/changelog',
    '/blog/:path*', '/careers/:path*', '/docs', '/help', '/status', '/security',
    '/use-cases', '/partners', '/faq', '/setup-keys', '/pricing', '/contact', '/feedback',
  ],
};
