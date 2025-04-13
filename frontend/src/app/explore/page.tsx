'use client'

import BoroughCard from '@/components/borough-card'
import { Loader } from '@/components/loader'
import { useBoroughsContext } from '@/context/boroughs-context'

export default function ExplorePage() {
  const { boroughs, loading } = useBoroughsContext()

  if (loading) {
    return (
      <div className="mt-8 flex items-center justify-center gap-3 text-gray-400">
        <span>Loading boroughs...</span>
        <Loader />
      </div>
    )
  }

  return (
    <div className="max-w-9xl mx-auto py-10">
      <h1 className="mb-6 text-center text-3xl font-bold text-white">
        Explore All Boroughs
      </h1>

      <div className="grid grid-cols-1 gap-6 px-4 sm:grid-cols-2 md:grid-cols-3">
        {boroughs?.map((borough) => (
          <BoroughCard key={borough.slug} {...borough} />
        ))}
      </div>
    </div>
  )
}
