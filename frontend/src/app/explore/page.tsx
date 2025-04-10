'use client'

import BoroughCard from '@/components/borough-card'
import { useBoroughsContext } from '@/context/boroughs-context'

export default function ExplorePage() {
  const { boroughs, loading, error } = useBoroughsContext()

  if (loading) {
    return <div className="text-center text-gray-400">Loading boroughs...</div>
  }

  if (error || !boroughs) {
    return (
      <div className="text-center text-red-400">Failed to load boroughs.</div>
    )
  }

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
