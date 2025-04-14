from borough.models import Borough

# Slug → image path (relative to Cloudinary)
image_paths = {
    "barnet": "boroughs/Friary_Gardens_Cathays_Park_11_May_2021_pgkmeh",
    "bexley": "boroughs/bexley_teelem",
    "brent": "boroughs/15445780165_8586f469e2_b_suyihq",
    "bromley": "boroughs/SpotBromley-4_tihw7n",
    "camden": "boroughs/Camden_town_1_kj6ex6",
    "city-of-london": "boroughs/2016-02_City_of_London_xwvrtd",
    "croydon": "boroughs/image_zhsdax",
    "ealing": "boroughs/Ealing-IMAGE3-CB_l9n3vk",
    "enfield": "boroughs/New_River_Loop_Enfield_Town_6522696317_dvvjk7",
    "greenwich": "boroughs/free-photo-of-greenwich-landscape_bxtbxv",
    "hackney": "boroughs/2542816_f1d9d03c_i0nr7l",
    "hammersmith-and-fulham": "boroughs/2056200_cd44bbb6_wg9aop",
    "islington": "boroughs/6553558_973d213a_sjrfym",
    "kensington-and-chelsea": "boroughs/0_iiGlTiv6vg1Mm6R0_ycxx7f",
    "lambeth": "boroughs/wagtailsource-lambeth-bridge.height-1200.format-webp_fteghp",
    "redbridge": "boroughs/The_Thames_Riverside_At_Richmond_-_London._14344421406_zbjh82",
    "southwark": "boroughs/e3bcc001455a4d3a5989cd15b1b44af2_1_h2pd4z",
    "sutton": "boroughs/25198419819_cb78fa1f81_b_c7ot0p",
    "tower-hamlets": "boroughs/The_White_Tower._City_of_London_qugpld",
    "waltham-forest": "boroughs/waltham_forest_2_brhsp1",
    "wandsworth": "boroughs/wandsworth_gtfpy7",
    "westminster": "boroughs/london-velikobritaniia-big-ben-westminster-bridge_1_so4esk",
    "haringey": "boroughs/haringey_wmr0po",
    "newham": "boroughs/newham_masuf6",
    "lewisham": "boroughs/lewisham_vzqjmi",
    "barking-and-dagenham": "barking_and_dagenham_pxbcs1",
    "hillingdon": "hillingdon_cmpg91",
}

BASE_URL = "https://res.cloudinary.com/djmajnlwm/image/upload/v1"

updated = 0
for borough in Borough.objects.all():
    image_rel_path = image_paths.get(borough.slug)
    if image_rel_path:
        borough.image = f"{BASE_URL}/{image_rel_path}"
        borough.save()
        updated += 1
