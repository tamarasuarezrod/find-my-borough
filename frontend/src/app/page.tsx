'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import BoroughCard from '@/components/borough-card';

type Borough = {
  name: string;
  slug: string;
  image: string; 
};

export default function HomePage() {
  const [boroughs, setBoroughs] = useState<Borough[]>([]);

  useEffect(() => {
    fetch('http://127.0.0.1:8000/api/boroughs/')
      .then((res) => res.json())
      .then((data) => {
        const featured = data.filter((b: Borough) =>
          ['camden', 'westminster', 'hackney'].includes(b.slug)
        );
        setBoroughs(featured);
      })
      .catch((err) => console.error('Failed to fetch boroughs:', err));
  }, []);

  return (
    <div className="text-center">
      <div
        className="relative h-[60vh] flex flex-col justify-center items-center text-white"
        style={{
          backgroundImage: 'url(/images/london-background.png)',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        <h2 className="text-3xl md:text-5xl font-bold mb-4">Find the perfect borough to live in London</h2>
        <div className="flex gap-4 mt-4">
          <Link href="/match" className="bg-white text-black px-6 py-2 rounded-full shadow hover:scale-105 transition">
            ✨ Get your match
          </Link>
          <span className="self-center text-lg">or</span>
          <Link href="/explore" className="bg-transparent border border-white px-6 py-2 rounded-full hover:bg-white hover:text-black transition">
            🔍 Explore
          </Link>
        </div>
      </div>

     { <div className="grid grid-cols-1 sm:grid-cols-3 gap-6 px-6 py-10">
        {boroughs.map((b) => (
          <BoroughCard
            key={b.slug}
            slug={b.slug}
            name={b.name}
            image={b.image}
          />
        ))}
      </div>}
    </div>
  );
}
