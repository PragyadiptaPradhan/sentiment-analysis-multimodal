import { type NextAuthConfig } from "next-auth";

/**
 * Edge-compatible auth config (no Prisma adapter or DB-dependent providers).
 * Used by middleware which runs in the Edge Runtime.
 */
export const edgeAuthConfig = {
  pages: {
    signIn: "/login",
  },
  providers: [],
  session: {
    strategy: "jwt",
  },
  callbacks: {
    session: ({ session, token }) => ({
      ...session,
      user: {
        ...session.user,
        id: token.sub,
      },
    }),
  },
} satisfies NextAuthConfig;
