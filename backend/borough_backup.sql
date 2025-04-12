--
-- PostgreSQL database dump
--

-- Dumped from database version 14.15 (Homebrew)
-- Dumped by pg_dump version 14.15 (Homebrew)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Data for Name: borough_borough; Type: TABLE DATA; Schema: public; Owner: tamarasuarez
--

COPY public.borough_borough (id, name, slug, image, norm_rent, norm_crime, norm_youth, norm_centrality) FROM stdin;
14	haringey	haringey		0.7094736842105263	0.7450200834608326	0.43009789148236516	0.33333333333333337
15	harrow	harrow		0.7719298245614035	0.9412126017667489	0.23317633997749507	0
16	havering	havering		0.9852631578947368	0.882611333847053	0.16631966249888583	0
17	hillingdon	hillingdon		0.8771929824561404	0.7828008145062522	0.24717460220613469	0
18	hounslow	hounslow		0.7287719298245614	0.7929523618075487	0.27736206255494766	0
19	islington	islington	boroughs/6553558_973d213a_sjrfym	0.47122807017543855	0.7546728015218014	0.8497174866174523	0.6666666666666667
20	kensington and chelsea	kensington-and-chelsea	boroughs/0_iiGlTiv6vg1Mm6R0_ycxx7f	0	0.8523912161382428	0.47362974071725267	1
21	kingston upon thames	kingston-upon-thames		0.736842105263158	1	0.22368687005659024	0
22	lambeth	lambeth	boroughs/wagtailsource-lambeth-bridge.height-1200.format-webp_fteghp	0.6157894736842106	0.653898127156785	0.8402751544031536	0.6666666666666667
23	lewisham	lewisham		0.8947368421052632	0.7470154004221435	0.4580593396277998	0.33333333333333337
24	merton	merton		0.6768421052631579	0.9682461685074955	0.29556297398218656	0
1	barking and dagenham	barking-and-dagenham		0.964561403508772	0.8754713750190783	0.25812641441887396	0
2	barnet	barnet	boroughs/Friary_Gardens_Cathays_Park_11_May_2021_pgkmeh	0.7094736842105263	0.7582390583295177	0.23357935317853565	0
3	bexley	bexley	boroughs/bexley_teelem	1	0.9338753448064059	0.1326361703686969	0
4	brent	brent	boroughs/15445780165_8586f469e2_b_suyihq	0.6743859649122808	0.7204434368590138	0.41068718509681235	0.33333333333333337
25	newham	newham		0.843859649122807	0.6280557944227914	0.5963634757620888	0.33333333333333337
26	redbridge	redbridge		0.9449122807017544	0.8135607101243723	0.2534025408672813	0
27	richmond upon thames	richmond-upon-thames	boroughs/The_Thames_Riverside_At_Richmond_-_London._14344421406_zbjh82	0.5614035087719298	0.9983843888783415	0	0
28	southwark	southwark	boroughs/e3bcc001455a4d3a5989cd15b1b44af2_1_h2pd4z	0.608421052631579	0.6383376329435765	0.7449317414776027	0.6666666666666667
5	bromley	bromley	boroughs/SpotBromley-4_tihw7n	0.8771929824561404	0.8255586701361357	0.057960162590459455	0
29	sutton	sutton	boroughs/25198419819_cb78fa1f81_b_c7ot0p	0.9385964912280702	0.9766815943178138	0.09936060455934717	0
6	camden	camden	boroughs/Camden_town_1_kj6ex6	0.38035087719298244	0.6212769284031137	0.6442006874182449	1
7	city of london	city-of-london	boroughs/2016-02_City_of_London_xwvrtd	0.3119298245614035	\N	0.9686978933212819	1
30	tower hamlets	tower-hamlets	boroughs/The_White_Tower._City_of_London_mt8jek	0.5378947368421052	0.66330143059759	1	0.6666666666666667
31	waltham forest	waltham-forest	boroughs/waltham_forest_2_brhsp1	0.8947368421052632	0.8273380759337228	0.37892188379885733	0.33333333333333337
8	croydon	croydon	boroughs/image_zhsdax	0.9473684210526316	0.671327369718087	0.23203504893001023	0
9	ealing	ealing	boroughs/Ealing-IMAGE3-CB_l9n3vk	0.7017543859649122	0.7237416660152105	0.31061275995258864	0.33333333333333337
10	enfield	enfield	boroughs/New_River_Loop_Enfield_Town_6522696317_dvvjk7	0.8245614035087719	0.7336810247590544	0.16682172717615373	0
11	greenwich	greenwich	boroughs/free-photo-of-greenwich-landscape_bxtbxv	0.7807017543859649	0.7710559917209237	0.40608393185567426	0.33333333333333337
12	hackney	hackney	boroughs/2542816_f1d9d03c_i0nr7l	0.5778947368421052	0.6984391111905267	0.7263590591646117	0.6666666666666667
13	hammersmith and fulham	hammersmith-and-fulham	boroughs/2056200_cd44bbb6_wg9aop	0.5094736842105263	0.8636074288330746	0.7690956299954022	0.6666666666666667
32	wandsworth	wandsworth	boroughs/wandsworth_gtfpy7	0.5385964912280702	0.7936447665739738	0.8120471280839539	0.6666666666666667
33	westminster	westminster	boroughs/london-velikobritaniia-big-ben-westminster-bridge_1_so4esk	0.1596491228070176	0	0.7002505568694541	1
\.


--
-- Name: borough_borough_id_seq; Type: SEQUENCE SET; Schema: public; Owner: tamarasuarez
--

SELECT pg_catalog.setval('public.borough_borough_id_seq', 37, true);


--
-- PostgreSQL database dump complete
--

