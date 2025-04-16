import NextAuth, { NextAuthOptions } from 'next-auth'
import FacebookProvider from 'next-auth/providers/facebook'
import GoogleProvider from 'next-auth/providers/google'

declare module 'next-auth' {
  interface Session {
    id_token?: string
  }

  interface JWT {
    id_token?: string
  }
}

const authOptions: NextAuthOptions = {
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID!,
      clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    }),
    FacebookProvider({
      clientId: process.env.FACEBOOK_CLIENT_ID!,
      clientSecret: process.env.FACEBOOK_CLIENT_SECRET!,
    }),
  ],
  callbacks: {
    async jwt({ token, account }) {
      if (account?.provider === 'google') {
        token.id_token = account.id_token
      }
      return token
    },
    async session({ session, token }) {
      session.id_token = token.id_token as string | undefined
      return session
    },
  },
}

const handler = NextAuth(authOptions)

export { handler as GET, handler as POST }
