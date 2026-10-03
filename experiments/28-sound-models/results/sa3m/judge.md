# Машинный отбор

| элемент | выбран | CLAP | PQ | забраковано |
|---|---|---|---|---|
| winter_outside | f14446s11 | 0.514 | 5.32 | 2 из 6 |
| hut_air | 39b9acs14 | 0.232 | 5.27 | 0 из 4 |
| fire_burning | d0ee6as13 | 0.520 | 5.77 | 0 из 4 |
| door_open | 8e0cc2s12 ⚠ все плохи, взят наименее плохой | 0.292 | 7.28 | 8 из 8 |
| door_close | 1c79f4s18 | 0.362 | 6.84 | 6 из 8 |
| floor_walk | 77f2aas11 ⚠ все плохи, взят наименее плохой | 0.354 | 6.96 | 8 из 8 |
| table_thud | 9565fas11 | 0.167 | 6.51 | 1 из 4 |
| cough | fc91c4s11 | 0.346 | 7.56 | 0 из 4 |
| fire_catching | d10782s15 ⚠ все плохи, взят наименее плохой | 0.065 | 6.91 | 8 из 8 |

## winter_outside — цель: fierce winter wind and blizzard outside | strong winter blizzard wind blowing
- ✔ `winter_outside__f14446s11` CLAP 0.514, PQ 5.32, RMS 0.083 — годен. AST: Field recording 0.34, Vehicle 0.19, Rumble 0.14
- ✘ `winter_outside__f14446s14` CLAP 0.491, PQ 5.16, RMS 0.132 — AST: engine 0.22. AST: Field recording 0.24, Vehicle 0.22, Wind noise (microphone) 0.08
- ✘ `winter_outside__f14446s13` CLAP 0.453, PQ 5.31, RMS 0.095 — AST: engine 0.25. AST: Wind 0.26, Vehicle 0.25, Wind noise (microphone) 0.23
- · `winter_outside__f14446s12` CLAP 0.440, PQ 5.10, RMS 0.100 — годен. AST: Field recording 0.14, Vehicle 0.12, Wind noise (microphone) 0.11
- · `winter_outside__dsp21` CLAP 0.385, PQ 5.89, RMS 0.065 — годен. AST: Eruption 0.12, White noise 0.07, Waves, surf 0.06
- · `winter_outside__dsp22` CLAP 0.373, PQ 5.79, RMS 0.050 — годен. AST: Eruption 0.13, White noise 0.07, Field recording 0.07

## hut_air — цель: still cold air in a wooden room | the quiet interior of a room
- ✔ `hut_air__39b9acs14` CLAP 0.232, PQ 5.27, RMS 0.092 — годен. AST: Computer keyboard 0.10, Vehicle 0.07, Typing 0.07
- · `hut_air__39b9acs13` CLAP 0.132, PQ 5.74, RMS 0.029 — годен. AST: Bouncing 0.12, Cupboard open or close 0.11, Door 0.10
- · `hut_air__39b9acs11` CLAP 0.110, PQ 6.37, RMS 0.031 — годен. AST: Thunk 0.20, Shuffling cards 0.07, Tap 0.07
- · `hut_air__39b9acs12` CLAP 0.114, PQ 5.58, RMS 0.024 — годен. AST: Crushing 0.21, Thunk 0.08, Tap 0.05

## fire_burning — цель: wood fire crackling in a stove | a wood fire crackling
- ✔ `fire_burning__d0ee6as13` CLAP 0.520, PQ 5.77, RMS 0.022 — годен. AST: Finger snapping 0.44, Vehicle 0.09, Wind 0.06
- · `fire_burning__d0ee6as12` CLAP 0.520, PQ 5.75, RMS 0.021 — годен. AST: Finger snapping 0.15, Vehicle 0.07, Crushing 0.06
- · `fire_burning__d0ee6as14` CLAP 0.510, PQ 5.84, RMS 0.025 — годен. AST: Finger snapping 0.28, Vehicle 0.06, Crushing 0.05
- · `fire_burning__d0ee6as11` CLAP 0.496, PQ 5.88, RMS 0.028 — годен. AST: Finger snapping 0.33, Wind 0.10, Fire 0.07

## door_open — цель: heavy wooden door creaking open; ждём: Door, Creak, Squeak
- ✘ `door_open__8e0cc2s11` CLAP 0.460, PQ 6.92, RMS 0.102 — AST: animal 0.35; AST: слышно Meow 0.35, а Door только 0.02. AST: Meow 0.35, Cat 0.17, Speech 0.15
- ✘ `door_open__8e0cc2s18` CLAP 0.444, PQ 7.02, RMS 0.102 — AST: слышно Speech 0.20, а Door только 0.01. AST: Speech 0.20, Music 0.08, Sound effect 0.07
- ✘ `door_open__8e0cc2s13` CLAP 0.340, PQ 7.15, RMS 0.119 — AST: слышно Vehicle 0.08, а Creak только 0.01. AST: Vehicle 0.08, Sound effect 0.06, Music 0.05
- ✘ `door_open__8e0cc2s15` CLAP 0.332, PQ 7.49, RMS 0.112 — AST: слышно Music 0.06, а Door только 0.01. AST: Sound effect 0.13, Music 0.06, Zipper (clothing) 0.04
- ✘ `door_open__8e0cc2s14` CLAP 0.337, PQ 6.82, RMS 0.118 — AST: слышно Vehicle 0.14, а Creak только 0.02. AST: Vehicle 0.14, Clatter 0.07, Accelerating, revving, vroom 0.06
- ✘ `door_open__8e0cc2s16` CLAP 0.301, PQ 7.19, RMS 0.107 — AST: слышно Zipper (clothing) 0.07, а Creak только 0.02. AST: Zipper (clothing) 0.07, Sound effect 0.03, Rattle 0.03
- ✔ `door_open__8e0cc2s12` CLAP 0.292, PQ 7.28, RMS 0.104 — AST: слышно Rattle 0.06, а Creak только 0.02. AST: Rattle 0.06, Clatter 0.06, Sewing machine 0.03
- ✘ `door_open__8e0cc2s17` CLAP 0.279, PQ 7.45, RMS 0.114 — AST: слышно Rattle 0.07, а Creak только 0.01. AST: Sound effect 0.18, Rattle 0.07, Scrape 0.05

## door_close — цель: heavy wooden door creaking shut; ждём: Door, Creak, Thump, thud
- ✔ `door_close__1c79f4s18` CLAP 0.362, PQ 6.84, RMS 0.060 — годен. AST: Creak 0.25, Sound effect 0.16, Scrape 0.09
- ✘ `door_close__1c79f4s13` CLAP 0.333, PQ 7.21, RMS 0.079 — AST: слышно Zipper (clothing) 0.34, а Creak только 0.01. AST: Zipper (clothing) 0.34, Sound effect 0.32, Fart 0.09
- ✘ `door_close__1c79f4s12` CLAP 0.337, PQ 6.91, RMS 0.063 — AST: слышно Fart 0.05, а Creak только 0.01. AST: Sound effect 0.23, Fart 0.05, Scratch 0.04
- · `door_close__1c79f4s15` CLAP 0.328, PQ 7.04, RMS 0.088 — годен. AST: Zipper (clothing) 0.18, Sound effect 0.15, Creak 0.11
- ✘ `door_close__1c79f4s14` CLAP 0.318, PQ 6.98, RMS 0.079 — AST: слышно Zipper (clothing) 0.42, а Creak только 0.10. AST: Zipper (clothing) 0.42, Sound effect 0.13, Creak 0.10
- ✘ `door_close__1c79f4s17` CLAP 0.302, PQ 6.11, RMS 0.089 — AST: слышно Zipper (clothing) 0.21, а Creak только 0.01. AST: Zipper (clothing) 0.21, Fart 0.17, Sound effect 0.16
- ✘ `door_close__1c79f4s16` CLAP 0.238, PQ 7.50, RMS 0.064 — AST: слышно Creak 0.07, а Creak только 0.07. AST: Creak 0.07, Zipper (clothing) 0.06, Inside, small room 0.03
- ✘ `door_close__1c79f4s11` CLAP 0.156, PQ 6.56, RMS 0.086 — AST: слышно Fart 0.31, а Creak только 0.05. AST: Fart 0.31, Zipper (clothing) 0.29, Sound effect 0.12

## floor_walk — цель: footsteps on wooden floorboards; ждём: Walk, footsteps, Creak
- ✘ `floor_walk__77f2aas14` CLAP 0.370, PQ 6.86, RMS 0.038 — AST: слышно Thunk 0.23, а Walk, footsteps только 0.01. AST: Thunk 0.23, Door 0.19, Tap 0.08
- ✔ `floor_walk__77f2aas11` CLAP 0.354, PQ 6.96, RMS 0.044 — AST: слышно Crackle 0.10, а Walk, footsteps только 0.02. AST: Crackle 0.10, Crushing 0.06, Thunk 0.06
- ✘ `floor_walk__77f2aas17` CLAP 0.345, PQ 7.27, RMS 0.039 — AST: слышно Thunk 0.12, а Walk, footsteps только 0.01. AST: Thunk 0.12, Crackle 0.07, Tap 0.06
- ✘ `floor_walk__77f2aas18` CLAP 0.335, PQ 7.31, RMS 0.033 — AST: слышно Crackle 0.10, а Walk, footsteps только 0.03. AST: Crackle 0.10, Tap 0.07, Inside, small room 0.04
- ✘ `floor_walk__77f2aas16` CLAP 0.330, PQ 7.14, RMS 0.050 — AST: слышно Thunk 0.30, а Walk, footsteps только 0.03. AST: Thunk 0.30, Sound effect 0.07, Crack 0.06
- ✘ `floor_walk__77f2aas13` CLAP 0.320, PQ 6.87, RMS 0.042 — AST: слышно Crackle 0.11, а Walk, footsteps только 0.05. AST: Crackle 0.11, Crushing 0.10, Thunk 0.05
- ✘ `floor_walk__77f2aas15` CLAP 0.306, PQ 7.28, RMS 0.043 — AST: слышно Crackle 0.14, а Walk, footsteps только 0.01. AST: Crackle 0.14, Door 0.06, Crushing 0.05
- ✘ `floor_walk__77f2aas12` CLAP 0.122, PQ 7.47, RMS 0.033 — AST: слышно Crackle 0.19, а Walk, footsteps только 0.01; CLAP: «rustling of dry hay» 0.95. AST: Crackle 0.19, Crunch 0.08, Crumpling, crinkling 0.07

## table_thud — цель: heavy object thudding on a table; ждём: Thump, thud, Thunk, Knock
- ✔ `table_thud__9565fas11` CLAP 0.167, PQ 6.51, RMS 0.085 — годен. AST: Knock 0.32, Door 0.07, Wood 0.04
- · `table_thud__9565fas12` CLAP 0.140, PQ 6.60, RMS 0.062 — годен. AST: Knock 0.43, Thunk 0.16, Music 0.04
- · `table_thud__9565fas14` CLAP 0.126, PQ 6.23, RMS 0.075 — годен. AST: Knock 0.10, Chopping (food) 0.04, Whack, thwack 0.04
- ✘ `table_thud__9565fas13` CLAP 0.121, PQ 5.81, RMS 0.062 — AST: слышно Wood 0.09, а Thunk только 0.02. AST: Wood 0.09, Sound effect 0.04, Dishes, pots, and pans 0.03

## cough — цель: deep rasping cough; ждём: Cough, Throat clearing
- ✔ `cough__fc91c4s11` CLAP 0.346, PQ 7.56, RMS 0.086 — годен. AST: Throat clearing 0.61, Cough 0.22, Speech 0.01
- · `cough__fc91c4s12` CLAP 0.350, PQ 6.91, RMS 0.082 — годен. AST: Throat clearing 0.73, Cough 0.08, Speech 0.01
- · `cough__fc91c4s14` CLAP 0.349, PQ 6.56, RMS 0.088 — годен. AST: Throat clearing 0.50, Cough 0.10, Speech 0.01
- · `cough__fc91c4s13` CLAP 0.339, PQ 6.10, RMS 0.066 — годен. AST: Throat clearing 0.81, Cough 0.05, Speech 0.02

## fire_catching — цель: fire starting and growing; ждём: Fire, Crackle
- ✘ `fire_catching__d10782s11` CLAP 0.172, PQ 7.40, RMS 0.012 — AST: слышно Crack 0.32, а Crackle только 0.02; CLAP: «rustling of dry hay» 1.00. AST: Crack 0.32, Crumpling, crinkling 0.08, Crushing 0.06
- ✘ `fire_catching__d10782s13` CLAP 0.175, PQ 6.69, RMS 0.018 — AST: слышно Crumpling, crinkling 0.38, а Crackle только 0.03; CLAP: «rustling of dry hay» 1.00. AST: Crumpling, crinkling 0.38, Inside, small room 0.31, Tearing 0.26
- ✘ `fire_catching__d10782s14` CLAP 0.108, PQ 7.41, RMS 0.016 — AST: слышно Tearing 0.25, а Crackle только 0.02; CLAP: «rustling of dry hay» 0.99. AST: Inside, small room 0.31, Tearing 0.25, Crumpling, crinkling 0.23
- ✘ `fire_catching__d10782s12` CLAP 0.104, PQ 6.79, RMS 0.018 — AST: слышно Crumpling, crinkling 0.39, а Crackle только 0.02; CLAP: «rustling of dry hay» 1.00. AST: Crumpling, crinkling 0.39, Inside, small room 0.30, Tearing 0.26
- ✘ `fire_catching__d10782s18` CLAP 0.096, PQ 7.13, RMS 0.021 — AST: слышно Crack 0.32, а Crackle только 0.01; CLAP: «rustling of dry hay» 1.00. AST: Crack 0.32, Crumpling, crinkling 0.18, Inside, small room 0.16
- ✘ `fire_catching__d10782s17` CLAP 0.094, PQ 6.87, RMS 0.021 — AST: слышно Crumpling, crinkling 0.44, а Crackle только 0.03; CLAP: «rustling of dry hay» 1.00. AST: Crumpling, crinkling 0.44, Tearing 0.39, Inside, small room 0.36
- ✔ `fire_catching__d10782s15` CLAP 0.065, PQ 6.91, RMS 0.019 — AST: слышно Crack 0.30, а Crackle только 0.01; CLAP: «rustling of dry hay» 0.92. AST: Crack 0.30, Crumpling, crinkling 0.20, Inside, small room 0.17
- ✘ `fire_catching__d10782s16` CLAP 0.060, PQ 7.03, RMS 0.020 — AST: слышно Tearing 0.60, а Crackle только 0.02; CLAP: «rustling of dry hay» 1.00. AST: Tearing 0.60, Crumpling, crinkling 0.34, Inside, small room 0.25
