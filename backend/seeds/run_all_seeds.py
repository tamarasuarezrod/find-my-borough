# python manage.py shell < seeds/run_all_seeds.py

print("Seeding the database...")

exec(open("seeds/seed_borough_features.py").read())
exec(open("seeds/seed_match_questions.py").read())
exec(open("seeds/seed_community_features.py").read())
exec(open("seeds/seed_borough_images.py").read())

print("🌱 All seeds loaded successfully")
