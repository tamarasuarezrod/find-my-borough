'use client'

import BoroughCard from '@/components/borough-card'
import { useEffect, useState } from 'react'

type Borough = {
  name: string
  slug: string
  image: string
}

export default function ExplorePage() {
  const [boroughs, setBoroughs] = useState<Borough[]>([])

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/boroughs/')
      .then((res) => res.json())
      .then((data) => setBoroughs(data))
      .catch((err) => console.error('Error fetching boroughs:', err))
  }, [])

  return (
    <div className="mx-auto max-w-7xl py-10">
      <h1 className="mb-6 text-center text-3xl font-bold text-white">
        Explore All Boroughs
      </h1>

      <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:grid-cols-3">
        {boroughs.map((borough) => (
          <BoroughCard key={borough.slug} {...borough} />
        ))}
      </div>
    </div>
  )
}
