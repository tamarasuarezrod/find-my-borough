'use client'

import Link from 'next/link'
import BoroughCard from '@/components/borough-card'
import { useBoroughs } from '@/services/get-boroughs'

export default function HomePage() {
  const { data: boroughs } = useBoroughs()

  return (
    <div className="text-center">
      <div
        className="relative flex h-[90vh] flex-col items-center justify-center text-white"
        style={{
          backgroundImage: 'url(/images/london-background.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        <h2 className="mb-4 text-3xl md:text-5xl">
          Find the perfect borough to live in London
        </h2>
        <div className="mt-4 flex flex-col items-center gap-4 sm:flex-row">
          <Link
            href="/match"
            className="w-full rounded-full bg-white px-6 py-2 text-black shadow transition hover:scale-105 sm:w-auto"
          >
            ✨ Get your match
          </Link>
          <span className="text-lg sm:self-center">or</span>
          <Link
            href="/explore"
            className="w-full rounded-full border border-white bg-transparent px-6 py-2 transition hover:bg-white hover:text-black sm:w-auto"
          >
            🔍 Explore
          </Link>
        </div>
      </div>

      {
        <div className="grid grid-cols-1 gap-6 px-6 py-10 sm:grid-cols-3">
          {boroughs
            ?.filter((b) =>
              ['camden', 'westminster', 'hackney'].includes(b.slug),
            )
            .map((borough) => (
              <BoroughCard
                key={borough.slug}
                slug={borough.slug}
                name={borough.name}
                image={borough.image}
              />
            ))}
        </div>
      }
    </div>
  )
}
