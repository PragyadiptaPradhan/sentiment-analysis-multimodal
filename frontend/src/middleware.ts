import NextAuth from "next-auth";
import { edgeAuthConfig } from "./server/auth/config.edge";

const { auth } = NextAuth(edgeAuthConfig);

export default auth((req) => {
    const isAuthenticated = !!req.auth;

    if (!isAuthenticated) {
        const newUrl = new URL("/login", req.nextUrl.origin);

        return Response.redirect(newUrl);
    }
});

export const config = {
    matcher: ["/"],
};
