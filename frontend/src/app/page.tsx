'use client'

import { useEffect, useState } from 'react'
import Link from 'next/link'
import BoroughCard from '@/components/borough-card'

type Borough = {
  name: string
  slug: string
  image: string
}

export default function HomePage() {
  const [boroughs, setBoroughs] = useState<Borough[]>([])

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/boroughs/')
      .then((res) => res.json())
      .then((data) => {
        const featured = data.filter((b: Borough) =>
          ['camden', 'westminster', 'hackney'].includes(b.slug),
        )
        setBoroughs(featured)
      })
      .catch((err) => console.error('Failed to fetch boroughs:', err))
  }, [])

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
        <div className="mt-4 flex gap-4">
          <Link
            href="/match"
            className="rounded-full bg-white px-6 py-2 text-black shadow transition hover:scale-105"
          >
            ✨ Get your match
          </Link>
          <span className="self-center text-lg">or</span>
          <Link
            href="/explore"
            className="rounded-full border border-white bg-transparent px-6 py-2 transition hover:bg-white hover:text-black"
          >
            🔍 Explore
          </Link>
        </div>
      </div>

      {
        <div className="grid grid-cols-1 gap-6 px-6 py-10 sm:grid-cols-3">
          {boroughs.map((borough) => (
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
