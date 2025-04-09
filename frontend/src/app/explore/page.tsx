'use client';

import BoroughCard from '@/components/borough-card';
import { useEffect, useState } from 'react';

type Borough = {
  name: string;
  slug: string;
  image: string;
};

export default function ExplorePage() {
  const [boroughs, setBoroughs] = useState<Borough[]>([]);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/boroughs/')
      .then((res) => res.json())
      .then((data) => setBoroughs(data))
      .catch((err) => console.error('Error fetching boroughs:', err));
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      <h1 className="text-3xl font-bold mb-6 text-white text-center">Explore All Boroughs</h1>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
        {boroughs.map((b) => (
          <BoroughCard
            key={b.slug}
            slug={b.slug}
            name={b.name}
            image={b.image}
          />
        ))}
      </div>
    </div>
  );
}
