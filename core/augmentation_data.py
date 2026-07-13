"""
core/augmentation_data.py
=========================
PhiUSIIL's "legitimate" class is 100% bare-homepage URLs (verified
empirically). Zero legitimate training examples have
a real path, so the model quietly learned "any real path = phishing,"
which would misclassify most normal web traffic (articles, docs, product
pages - anything that isn't a homepage).

IMPORTANT - v2 (2026-07-07): the original ~69-URL version of this file
was proven, via a 100,000-URL evaluation, to cause MEMORIZATION rather
than generalization. Matched-pair evidence: a verbatim-augmented path on
pandas.pydata.org scored 0.22% phishing; a same-shape, unseen path on the
different-but-similar numpy.org scored 99.94%. Even worse: two different
paths on the SAME domain (realpython.com) - one augmented, one not -
scored 0.40% and 99.86% respectively. The model wasn't learning "this
kind of site is trustworthy," it was closer to memorizing literal
training strings.

The fix is DIVERSITY, not more replication of a small set. This version
pulls real URLs from ~150+ distinct domains across 13 genuinely different
topic categories (outdoor recreation, government services, health,
personal finance/mortgages, consumer tech reviews, sports news, online
education, insurance, real estate/rentals, and more from the original
v1 set), specifically so the model has to learn the general SHAPE of a
legitimate content URL rather than memorize a short list of specific
strings. Replication is correspondingly much lower (see
models/train.py) - high replication of a small set is the mechanism that
caused the memorization problem in the first place, not just an
efficiency question.

v3 (2026-07-09): the realistic held-out evaluation (models/evaluate.py)
still showed a 47% false-positive rate on legitimate content URLs -
every error was a benign page with a hyphenated multi-word path on a
domain/category the model had never seen benign. Expanded with ~105 more
real, search-sourced URLs across 12 additional categories (car
maintenance, gardening, museums, university admissions, music lessons,
travel visas, home DIY, chess, parenting, astronomy, cycling, coffee),
deliberately including non-.com TLDs (.edu, .org, .gov.uk, .co.uk,
.nhs.uk, .si.edu) that were underrepresented. IMPORTANT for anyone
adding more: these categories were chosen to stay DISJOINT from
models/evaluate.py's held-out categories (freelance taxes, puppy
training) - adding URLs from those categories would contaminate the
held-out measurement and make the numbers meaningless. Replication
multiplier unchanged (8x - see the memorization warning above).
"""

REAL_BENIGN_URLS_WITH_PATHS = [
    # --- Reference / encyclopedia ---
    "https://en.wikipedia.org/wiki/Wikipedia:About",
    "https://en.wikipedia.org/wiki/Wikipedia",
    "https://en.wikipedia.org/wiki/English_Wikipedia",
    "https://en.wikipedia.org/wiki/Wiki",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://en.wikipedia.org/wiki/Wikipedia:Contents",
    "https://en.wikipedia.org/wiki/List_of_Wikipedias",
    "https://en.wikipedia.org/wiki/Wikimedia_Foundation",
    "https://en.wikipedia.org/wiki/History_of_Wikipedia",
    "https://en.wikipedia.org/wiki/Digital_India",

    # --- Government / public services (India) ---
    "https://www.india.gov.in/",
    "https://dic.gov.in/",
    "https://services.india.gov.in/service/detail/skill-india-digital-hub",
    "https://www.digitalindia.gov.in/",
    "https://services.india.gov.in/",
    "https://guidelines.india.gov.in/head/national-government-services-portal-https-services-india-gov-in/",
    "https://services.india.gov.in/page/show/about_us/en",
    "https://www.digilocker.gov.in/",
    "https://csc.gov.in/digitalIndia",

    # --- Government / public services (US - passports) ---
    "https://travel.state.gov/en/passports/renew-replace/online.html",
    "https://travel.state.gov/en/passports/renew-replace.html",
    "https://www.usa.gov/passport",
    "https://www.usps.com/international/passports.htm",
    "https://travel.state.gov/en/passports/renew-replace/mail.html",
    "https://www.usa.gov/apply-adult-passport",

    # --- Cooking / lifestyle blogs ---
    "https://alexandracooks.com/2017/10/24/artisan-sourdough-made-simple-sourdough-bread-demystified-a-beginners-guide-to-sourdough-baking/",
    "https://www.theclevercarrot.com/2014/01/sourdough-bread-a-beginners-guide/",
    "https://www.theperfectloaf.com/beginners-sourdough-bread/",
    "https://pantrymama.com/how-to-bake-simple-sourdough-bread/",
    "https://littlespoonfarm.com/sourdough-bread-recipe-beginners-guide/",
    "https://www.farmhouseonboone.com/beginners-sourdough-bread-recipe/",
    "https://amybakesbread.com/easy-sourdough-bread-recipe/",
    "https://sugarspunrun.com/sourdough-bread-recipe/",

    # --- Developer docs / tutorials ---
    "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
    "https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-groupby/",
    "https://pandas.pydata.org/docs/user_guide/groupby.html",
    "https://realpython.com/pandas-groupby/",
    "https://www.datacamp.com/tutorial/pandas-groupby",
    "https://www.w3schools.com/python/pandas/ref_df_groupby.asp",
    "https://www.tutorialspoint.com/python_pandas/python_pandas_groupby.htm",
    "https://www.ionos.com/digitalguide/websites/web-development/python-pandas-dataframe-groupby/",
    "https://www.codecademy.com/resources/docs/pandas/dataframe/groupby",
    "https://www.programiz.com/python-programming/pandas/groupby",

    # --- Product reviews / commerce-adjacent content ---
    "https://www.whathifi.com/best-buys/headphones/best-noise-cancelling-headphones",
    "https://www.rtings.com/headphones/reviews/best/by-feature/noise-cancelling",
    "https://recordingnow.com/blog/best-noise-cancelling-headphones/",
    "https://www.audiophileon.com/news/best-noise-cancelling-headphones",
    "https://www.rollingstone.com/product-recommendations/tech/best-noise-canceling-headphones-1235398146/",
    "https://www.rtings.com/headphones/reviews/best/wireless-earbuds",
    "https://www.techgearlab.com/topics/audio/best-wireless-earbuds",
    "https://www.whathifi.com/best-buys/best-wireless-earbuds-budget-and-premium",
    "https://www.scarbir.com/guide/best-sounding-wireless-earphones-50-dollar",
    "https://www.loudnwireless.com/blog/2026-best-earbuds-budget-to-premium-kings",
    "https://www.crutchfield.com/learn/best-true-wireless-earbuds.html",

    # --- Code hosting / dev community ---
    "https://github.com/topics/trending-repositories",
    "https://github.com/mbadry1/Trending-Deep-Learning",
    "https://github.com/trending",
    "https://github.com/topics/machine-learning-projects",
    "https://www.firecrawl.dev/blog/best-github-repos",
    "https://www.kdnuggets.com/10-github-repositories-for-machine-learning-projects",
    "https://www.projectpro.io/article/machine-learning-projects-on-github/465",

    # --- Outdoor recreation / travel content ---
    "https://thebigoutside.com/the-20-best-national-park-dayhikes/",
    "https://www.paulintheparks.com/the-most-epic-hikes-in-the-national-parks/",
    "https://explorewithalec.com/top-national-park-hikes/",
    "https://www.earthtrekkers.com/best-day-hikes-in-the-national-parks/",
    "https://lemap.co/blogs/news/the-15-best-hikes-in-u-s-national-parks-trails-you-ll-never-forget",
    "https://www.thewanderingqueen.com/best-hikes-in-national-parks/",
    "https://theresearchedtraveler.com/best-hike-in-each-national-park/",
    "https://www.backpacker.com/trips/trips-by-national-park/the-best-50-dayhikes-backpacking-trips-and-thru-hikes-in-the-national-parks/",
    "https://www.undercanvas.com/blog/top-10-national-park-hiking-trails/",

    # --- Health / medical information ---
    "https://my.clevelandclinic.org/health/diseases/15050-vitamin-d-vitamin-d-deficiency",
    "https://www.healthline.com/nutrition/vitamin-d-deficiency-symptoms",
    "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10377780/",
    "https://patient.info/bones-joints-muscles/osteoporosis-leaflet/vitamin-d-deficiency",
    "https://www.endocrinecenter.com/blog/10-unexpected-signs-of-a-vitamin-d-deficiency",
    "https://www.ncbi.nlm.nih.gov/books/NBK532266/",

    # --- Personal finance / mortgages ---
    "https://www.lendingtree.com/home/refinance/how-to-refinance-a-mortgage/",
    "https://www.citizensbank.com/learning/refinance-process.aspx",
    "https://www.nerdwallet.com/mortgages/learn/how-to-refinance-your-mortgage",
    "https://finance.yahoo.com/personal-finance/mortgages/article/refinance-mortgage-162831396.html",
    "https://themortgagereports.com/87622/how-to-refinance-mortgage-guide",
    "https://www.freedommortgage.com/learn/refinancing/how-to-refinance",
    "https://www.ncsecu.org/resources/learn/how-to-refinance",
    "https://www.loandepot.com/learning-center/home-refinance/10-step-mortgage-refinance-process-complete-guide-homeowners",
    "https://www.rocketmortgage.com/learn/how-does-refinancing-work",
    "https://www.morty.com/resources/mortgage-101/how-to-refinance-mortgage",

    # --- Sports news ---
    "https://nbarankings.theringer.com/the-offer-sheet",
    "https://www.espn.com/nba/trade-deadline/",
    "https://basketball.realgm.com/nba/news",
    "https://www.hoopsrumors.com/",
    "https://www.nbcsports.com/nba/trade-deadline",
    "https://www.hoopshype.com/rumors/",
    "https://basketball.realgm.com/nba/transactions/trade-deadline",
    "https://www.espn.com/nba/story/_/id/49135150/2026-nba-offseason-trade-grades-contracts-new-deals-rosters-teams",

    # --- Online education ---
    "https://pll.harvard.edu/subject/data-science",
    "https://www.coursera.org/browse/data-science",
    "https://www.coursera.org/professional-certificates/ibm-data-science",
    "https://www.mygreatlearning.com/data-science/free-courses",
    "https://www.coursera.org/courses?query=data+science&topic=Data+Science",
    "https://pll.harvard.edu/series/professional-certificate-data-science",
    "https://learn.org/courses-and-certificates/best-online-data-science-courses-with-certificates",
    "https://365datascience.com/courses/",
    "https://www.classcentral.com/subject/data-science",
    "https://www.monroeu.edu/academics/king-graduate-school/online-advanced-certificate-data-science",

    # --- Insurance ---
    "https://insurify.com/",
    "https://www.thezebra.com/",
    "https://www.nerdwallet.com/insurance/auto/car-insurance",
    "https://www.thezebra.com/auto-insurance/",
    "https://www.compare.com/",
    "https://www.experian.com/insurance/car-insurance-quotes/",
    "https://www.geico.com/auto-insurance/comparison/",
    "https://www.usnews.com/insurance/auto",
    "https://www.progressive.com/auto/",
    "https://www.progressive.com/auto/discounts/compare-car-insurance-rates/",

    # --- Real estate / rentals ---
    "https://www.apartments.com/",
    "https://www.trulia.com/rent/",
    "https://www.apartmentguide.com/",
    "https://www.zumper.com/",
    "https://www.apartmentlist.com/",
    "https://www.redfin.com/rentals",
    "https://www.rentcafe.com/",
    "https://www.apartments.com/near-me/apartments-for-rent/",
    "https://www.rentable.co/",
    "https://www.renthop.com/",

    # ================= v3 additions (2026-07-09) =================
    # All real, search-sourced. See module docstring: 12 new categories,
    # disjoint from models/evaluate.py's held-out categories.

    # --- Car maintenance / DIY auto ---
    "https://www.autozone.com/diy/motor-oil/easy-steps-to-change-your-oil",
    "https://www.castrol.com/en_us/united-states/home/learn/car-maintenance/how-to-change-your-engine-oil.html",
    "https://www.dummies.com/article/home-auto-hobbies/automotive/car-repair-maintenance/engines/how-to-change-your-cars-oil-202864/",
    "https://www.wyotech.edu/guides/how-to-change-car-oil-5-steps/",
    "https://www.mopar.com/en-us/blog/how-to-change-your-vehicle-engine-oil-a-step-by-step-mopar-oil-change-guide.html",
    "https://www.slashgear.com/1708004/how-change-your-oil-step-by-step-guide/",
    "https://engineauditor.com/how-to-change-car-engine-oil-at-home/",
    "https://www.truecar.com/blog/how-to-change-your-engine-oil/",
    "https://www.mymotorworld.com/blog/how-to-change-oil-in-a-car-my-motor-worlds-guide-to-changing-your-engine-oil-",

    # --- Gardening ---
    "https://www.themakermakes.com/blog/growing-tomatoes-for-beginners",
    "https://www.foodgardenlife.com/learn/grow-tomato-seeds",
    "https://www.epicgardening.com/grow-tomatoes-from-seed/",
    "https://www.gardeners.com/blogs/tomato-growing-articles/video-slideshow-growing-tomatoes-7902",
    "https://www.bbg.org/article/starting_tomatoes_from_seed",
    "https://www.almanac.com/starting-tomato-seeds-indoors",
    "https://savvygardening.com/growing-tomatoes-from-seed/",
    "https://scottsmiraclegro.com/en-us/learn/gardening/growing-tomatoes-how-to-grow-tomatoes-from-seeds.html",
    "https://www.creativevegetablegardener.com/starting-tomatoes-from-seed/",

    # --- Museums / cultural institutions (.org-heavy) ---
    "https://www.amnh.org/plan-your-visit",
    "https://www.mcny.org/visit",
    "https://www.metmuseum.org/plan-your-visit",
    "https://www.guggenheim.org/plan-your-visit",
    "https://intrepidmuseum.org/plan-your-visit/visitor-information",
    "https://www.ushmm.org/information/visit-the-museum/plan-your-visit",
    "https://californiamuseum.org/visit/",
    "https://nmaahc.si.edu/visit/plan-your-visit",
    "https://intrepidmuseum.org/plan-your-visit/visitor-information/tickets",

    # --- University admissions (.edu) ---
    "https://www.seattleu.edu/admissions-aid/undergraduate-admissions/first-year-admissions/how-to-apply/admission-requirements/",
    "https://educationusa.state.gov/your-5-steps-us-study/complete-your-application/undergraduate",
    "https://www.nyu.edu/admissions/undergraduate-admissions/how-to-apply.html",
    "https://www.uh.edu/undergraduate-admissions/apply/",
    "https://admissions.charlotte.edu/apply/first-year-students/application-requirements/",
    "https://admission.universityofcalifornia.edu/admission-requirements/",
    "https://admit.washington.edu/apply/first-year/",
    "https://www.sandiego.edu/admission-and-aid/undergraduate/apply/application-requirements.php",
    "https://admissions.ua.edu/apply/",
    "https://admission.ucla.edu/apply/first-year",

    # --- Music lessons / guitar ---
    "https://www.schoolofrock.com/resources/guitar/guitar-chords-for-beginners",
    "https://www.imusic-school.com/en/tools/guitar-chords/beginner/",
    "https://www.guitarnoise.com/lessons/absolute-beginner-part-1/",
    "https://www.soundguitarlessons.com/blog/how-to-learn-guitar-chords-1",
    "https://www.theguitarlesson.com/guitar-theory/",
    "https://www.justinguitar.com/beginner",
    "https://artiumacademy.com/blogs/what-is-guitar-music-theory-behind-notes-chords-scales/",
    "https://www.stringkick.com/blog-lessons/guitar-music-theory/",

    # --- Travel / visas (.gov.uk, .co.uk, .org.uk) ---
    "https://www.gov.uk/standard-visitor",
    "https://www.gov.uk/standard-visitor/apply-standard-visitor-visa",
    "https://www.gov.uk/browse/visas-immigration/tourist-short-stay-visas",
    "https://www.visitbritain.com/en/plan-your-trip/visa-and-immigration-information",
    "https://www.davidsonmorris.com/documents-required-for-uk-visitor-visa/",
    "https://www.ein.org.uk/blog/what-documents-are-required-when-applying-uk-visitor-visa",
    "https://reissedwards.co.uk/immigration/uk-visitor-visa/",
    "https://legalclarity.org/uk-tourist-visa-requirements-documents-and-how-to-apply/",

    # --- Home improvement / DIY plumbing ---
    "https://www.lowes.com/n/how-to/repair-a-leaky-faucet/",
    "https://www.familyhandyman.com/project/how-to-repair-a-kitchen-faucet/",
    "https://westernrooter.com/leaky-faucet-repair-diy-fixes-for-dripping-faucets/",
    "https://www.homedepot.com/c/ah/how-to-fix-a-leaky-faucet/9ba683603be9fa5395fab90ee6659fb",
    "https://bonfe.com/blog/diy-guide-fix-a-leaky-faucet-in-5-simple-steps/",
    "https://mintera.com/blogs/news/diy-easiest-way-to-fix-a-leaky-faucet",
    "https://www.edwardsenterprisescc.com/plumbing/leaky-faucet-repair-diy/",
    "https://www.poolesplumbing.com/blog/fix-a-leaky-faucet-without-calling-a-plumber/",
    "https://www.benjaminfranklinplumbing.com/doylestown/blog/2018/june/diy-fix-a-leaky-faucet-in-10-simple-steps/",

    # --- Chess / board games ---
    "https://www.chess.com/article/view/the-best-chess-openings-for-beginners",
    "https://shop.worldchess.com/blogs/news/chess-openings-for-beginners",
    "https://www.chess.com/article/view/study-plan-for-beginners-the-opening",
    "https://www.chessable.com/blog/10-chess-openings-for-beginners/",
    "https://northtexaschessacademy.com/chess-openings-for-beginners-guide/",
    "https://dwheeler.com/chess-openings/",
    "https://www.amazon.com/Chess-Openings-Beginners-Essential-Strategies/dp/1638076790",
    "https://chessiverse.com/blog/how-to-master-chess-opening-strategies-a-beginners-guide",
    "https://www.thechesswebsite.com/chess-openings/",

    # --- Parenting / toddler sleep (.nhs.uk included) ---
    "https://www.healthychildren.org/English/healthy-living/sleep/Pages/bedtime-trouble.aspx",
    "https://www.takingcarababies.com/blogs/sleep-schedules/toddler-nap-schedules-for-2-3-and-4-year-olds",
    "https://kidshealth.org/en/parents/sleep12yr.html",
    "https://www.bedslutonchildrenshealth.nhs.uk/sleep/healthy-sleep-routines/sleep-routines-for-toddlers-and-children/",
    "https://www.pampers.com/en-us/toddler/sleep/article/understanding-toddler-sleep",
    "https://www.metropediatrics.com/pediatric-blog/how-to-help-your-toddler-sleep/",
    "https://www.arestfulnight.com/blog/toddler-sleep-schedule",
    "https://www.heavensentsleep.com/blog/a-comprehensive-guide-to-toddler-sleep-schedules",
    "https://www.enfamil.com/articles/toddler-bedtime-routine/",
    "https://ducklingselc.com/blog/2025/02/toddler-sleep-schedule/",

    # --- Astronomy / stargazing ---
    "https://skyandtelescope.org/astronomy-resources/stargazing-basics/",
    "https://www.skyatnightmagazine.com/advice/skills/stargazing-top-tips",
    "https://www.planetary.org/articles/a-beginners-guide-to-stargazing",
    "https://telescopeguides.com/stargazing-101-an-introductory-guide/",
    "https://www.skyatnightmagazine.com/advice/astronomy-for-beginners",
    "https://exploringthenightsky.com/stargazing-for-beginners/",
    "https://cosmicpursuits.com/start-here-stargazing-basics/",
    "https://www.telescopeadvisor.com/how-to-start-stargazing/",
    "https://www.telescopr.com/tips-for-stargazing",

    # --- Cycling / bike maintenance ---
    "https://www.rei.com/learn/expert-advice/bike-maintenance.html",
    "https://www.rei.com/learn/series/intro-to-bike-maintenance",
    "https://365cycles.com/blogs/general/bike-chain-maintenance",
    "https://diybikefix.com/bike-chain-maintenance/",
    "https://www.rei.com/learn/expert-advice/bike-chain.html",
    "https://rouvy.com/blog/how-to-clean-bike-chain",
    "https://www.bikeradar.com/advice/workshop/how-to-clean-a-bike-chain",
    "https://www.schwinnbikes.com/blogs/compass/how-to-maintain-your-bike-chain",
    "https://mountainbikeinsider.com/mountain-bike-chain-maintenance/",
    "https://www.thecrucible.org/guides/bike-maintenance/repair-a-bike/",

    # --- Coffee brewing ---
    "https://handground.com/french-press-coffee-to-water-ratio-calculator",
    "https://beanbox.com/blog/how-to-use-a-french-press",
    "https://myhomebarista.com/guides/pour-over-vs-french-press/",
    "https://thecoffeeclubshop.com/blogs/barista-blog/french-press-coffee-ratios-how-to-brew-espresso-hacks",
    "https://craftcoffeespot.com/french-press-coffee/",
    "https://counterculturecoffee.com/blogs/counter-culture-coffee/coffee-basics-brewing-ratios",
    "https://espro.com/blogs/news/french-press-coffee-ratio",
    "https://coffeebrewshub.com/blog/coffee-to-water-ratio",
    "https://www.stonecreekcoffee.com/blogs/news/coffee-to-water-ratio-coffee-brew-ratio-guide",
    "https://coffeebros.com/blogs/coffee/the-perfect-pour-over-guide",
]

# Bare-root URLs WITH a trailing slash - PhiUSIIL's legitimate class has
# a second, sharper artifact beyond "no real path": 100% of its benign
# examples have EXACTLY 2 slashes (https://domain.com, never a trailing
# slash). A model trained on that alone treats "https://discord.com/"
# (vs "https://discord.com") as ~100% phishing on slash-count almost
# by itself. These entries target that specifically.
REAL_BENIGN_ROOT_URLS_WITH_TRAILING_SLASH = [
    "https://www.google.com/",
    "https://www.youtube.com/",
    "https://www.amazon.com/",
    "https://www.microsoft.com/",
    "https://www.apple.com/",
    "https://discord.com/",
    "https://www.reddit.com/",
    "https://www.linkedin.com/",
    "https://github.com/",
    "https://www.wikipedia.org/",
    "https://www.instagram.com/",
    "https://www.netflix.com/",
    "https://www.spotify.com/",
    "https://www.perplexity.ai/",
    "https://www.dropbox.com/",
    "https://www.adobe.com/",
    "https://www.salesforce.com/",
    "https://www.ibm.com/",
    "https://www.oracle.com/",
    "https://www.cloudflare.com/",
]
