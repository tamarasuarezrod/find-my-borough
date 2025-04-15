from borough.models import Borough

# Slug → image path (relative to Cloudinary)
image_paths = {
    "barnet": "Friary_Gardens_Cathays_Park_11_May_2021_pgkmeh",
    "bexley": "bexley_teelem",
    "brent": "15445780165_8586f469e2_b_suyihq",
    "bromley": "SpotBromley-4_tihw7n",
    "camden": "Camden_town_1_kj6ex6",
    "city-of-london": "2016-02_City_of_London_xwvrtd",
    "croydon": "image_zhsdax",
    "ealing": "Ealing-IMAGE3-CB_l9n3vk",
    "enfield": "New_River_Loop_Enfield_Town_6522696317_dvvjk7",
    "greenwich": "free-photo-of-greenwich-landscape_bxtbxv",
    "hackney": "2542816_f1d9d03c_i0nr7l",
    "hammersmith-and-fulham": "2056200_cd44bbb6_wg9aop",
    "islington": "6553558_973d213a_sjrfym",
    "kensington-and-chelsea": "0_iiGlTiv6vg1Mm6R0_ycxx7f",
    "lambeth": "wagtailsource-lambeth-bridge.height-1200.format-webp_fteghp",
    "redbridge": "redbridge_vucjpv",
    "southwark": "e3bcc001455a4d3a5989cd15b1b44af2_1_h2pd4z",
    "sutton": "25198419819_cb78fa1f81_b_c7ot0p",
    "tower-hamlets": "The_White_Tower._City_of_London_qugpld",
    "waltham-forest": "waltham_forest_2_brhsp1",
    "wandsworth": "wandsworth_gtfpy7",
    "westminster": "london-velikobritaniia-big-ben-westminster-bridge_1_so4esk",
    "haringey": "haringey_wmr0po",
    "newham": "newham_masuf6",
    "lewisham": "lewisham_vzqjmi",
    "barking-and-dagenham": "barking_and_dagenham_pxbcs1",
    "hillingdon": "hillingdon_cmpg91",
    "havering": "havering_bn5oto",
    "merton": "merton_lpkqg7",
    "hounslow": "hounslow_f5wutk",
    "richmond-upon-thames": "richmond-upon-thames__stqtvz",
    "harrow": "harrow_vzvsae",
    "kingston-upon-thames": "kingston-upon-thames_psyuev",
}

BASE_URL = "https://res.cloudinary.com/djmajnlwm/image/upload/v1/boroughs"

updated = 0
for borough in Borough.objects.all():
    image_rel_path = image_paths.get(borough.slug)
    if image_rel_path:
        borough.image = f"{BASE_URL}/{image_rel_path}"
        borough.save()
        updated += 1
