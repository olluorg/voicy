# Машинный отбор

| элемент | выбран | CLAP | PQ | забраковано |
|---|---|---|---|---|
| winter_outside | f14446s14 | 0.432 | 5.87 | 2 из 6 |
| hut_air | 39b9acs13 | 0.445 | 5.63 | 0 из 4 |
| fire_burning | d0ee6as13 | 0.536 | 6.71 | 0 из 4 |
| door_open | 42708ds12 | 0.386 | 5.68 | 7 из 8 |
| door_close | 903eb9s16 | 0.302 | 6.63 | 7 из 8 |
| floor_walk | 9e5a6ds11 ⚠ все плохи, взят наименее плохой | 0.136 | 7.39 | 8 из 8 |
| table_thud | 9565fas11 | 0.163 | 7.05 | 3 из 4 |
| cough | 3a7ac3s11 ⚠ все плохи, взят наименее плохой | 0.035 | 5.72 | 8 из 8 |
| fire_catching | 4a92b0s18 | 0.433 | 6.01 | 5 из 8 |

## winter_outside — цель: fierce winter wind and blizzard outside | strong winter blizzard wind blowing
- ✘ `winter_outside__f14446s11` CLAP 0.488, PQ 4.64, RMS 0.128 — AST: mic_wind 0.54. AST: Wind noise (microphone) 0.54, Wind 0.39, Vehicle 0.18
- ✔ `winter_outside__f14446s14` CLAP 0.432, PQ 5.87, RMS 0.130 — годен. AST: Field recording 0.37, Eruption 0.17, Rumble 0.11
- · `winter_outside__f14446s12` CLAP 0.410, PQ 6.08, RMS 0.071 — годен. AST: Eruption 0.45, Field recording 0.23, Fixed-wing aircraft, airplane 0.11
- ✘ `winter_outside__f14446s13` CLAP 0.441, PQ 4.31, RMS 0.164 — AST: mic_wind 0.63. AST: Wind noise (microphone) 0.63, Wind 0.42, Eruption 0.16
- · `winter_outside__dsp21` CLAP 0.385, PQ 5.89, RMS 0.065 — годен. AST: Eruption 0.12, White noise 0.07, Waves, surf 0.06
- · `winter_outside__dsp22` CLAP 0.373, PQ 5.79, RMS 0.050 — годен. AST: Eruption 0.13, White noise 0.07, Field recording 0.07

## hut_air — цель: still cold air in a wooden room | the quiet interior of a room
- ✔ `hut_air__39b9acs13` CLAP 0.445, PQ 5.63, RMS 0.016 — годен. AST: Rumble 0.49, Silence 0.12, White noise 0.07
- · `hut_air__39b9acs12` CLAP 0.412, PQ 5.58, RMS 0.012 — годен. AST: Silence 0.09, Hum 0.05, White noise 0.04
- · `hut_air__39b9acs11` CLAP 0.387, PQ 5.62, RMS 0.015 — годен. AST: Rumble 0.73, White noise 0.07, Vehicle 0.06
- · `hut_air__39b9acs14` CLAP 0.376, PQ 5.14, RMS 0.014 — годен. AST: Rumble 0.06, Silence 0.05, Cupboard open or close 0.04

## fire_burning — цель: wood fire crackling in a stove | a wood fire crackling
- ✔ `fire_burning__d0ee6as13` CLAP 0.536, PQ 6.71, RMS 0.014 — годен. AST: Computer keyboard 0.15, Finger snapping 0.09, Typing 0.04
- · `fire_burning__d0ee6as11` CLAP 0.525, PQ 6.20, RMS 0.015 — годен. AST: Fire 0.16, Crushing 0.08, Hands 0.06
- · `fire_burning__d0ee6as12` CLAP 0.515, PQ 6.45, RMS 0.012 — годен. AST: Crushing 0.17, Fire 0.15, Crack 0.07
- · `fire_burning__d0ee6as14` CLAP 0.499, PQ 6.28, RMS 0.015 — годен. AST: Crushing 0.08, Fire 0.07, Crack 0.03

## door_open — цель: heavy wooden door creaking open; ждём: Door, Creak, Squeak
- ✔ `door_open__42708ds12` CLAP 0.386, PQ 5.68, RMS 0.030 — годен. AST: Door 0.19, Creak 0.14, Sliding door 0.09
- ✘ `door_open__42708ds15` CLAP 0.338, PQ 5.54, RMS 0.104 — AST: слышно Music 0.08, а Door только 0.01. AST: Music 0.08, Sound effect 0.04, Musical instrument 0.03
- ✘ `door_open__42708ds13` CLAP 0.330, PQ 5.38, RMS 0.241 — AST: music 0.65; AST: слышно Music 0.65, а Door только 0.00. AST: Music 0.65, Musical instrument 0.28, Didgeridoo 0.12
- ✘ `door_open__42708ds11` CLAP 0.289, PQ 5.39, RMS 0.051 — AST: music 0.35; AST: слышно Music 0.35, а Door только 0.02. AST: Music 0.35, Musical instrument 0.14, Knock 0.06
- ✘ `door_open__42708ds16` CLAP 0.262, PQ 4.63, RMS 0.108 — AST: слышно Rumble 0.77, а Squeak только 0.00. AST: Rumble 0.77, Sound effect 0.15, Car 0.04
- ✘ `door_open__42708ds14` CLAP 0.233, PQ 4.93, RMS 0.079 — AST: слышно Knock 0.96, а Door только 0.12. AST: Knock 0.96, Door 0.12, Tap 0.06
- ✘ `door_open__42708ds18` CLAP 0.145, PQ 5.43, RMS 0.208 — AST: слышно Knock 0.97, а Door только 0.09. AST: Knock 0.97, Door 0.09, Tap 0.04
- ✘ `door_open__42708ds17` CLAP 0.114, PQ 4.93, RMS 0.232 — AST: слышно Knock 0.43, а Door только 0.13. AST: Knock 0.43, Door 0.13, Heart sounds, heartbeat 0.12

## door_close — цель: heavy wooden door creaking shut; ждём: Door, Creak, Thump, thud
- ✔ `door_close__903eb9s16` CLAP 0.302, PQ 6.63, RMS 0.067 — годен. AST: Knock 0.63, Door 0.41, Slam 0.21
- ✘ `door_close__903eb9s12` CLAP 0.281, PQ 6.84, RMS 0.039 — AST: слышно Knock 0.97, а Door только 0.23. AST: Knock 0.97, Door 0.23, Slam 0.08
- ✘ `door_close__903eb9s18` CLAP 0.271, PQ 6.86, RMS 0.172 — AST: слышно Knock 0.56, а Door только 0.25. AST: Knock 0.56, Door 0.25, Slam 0.16
- ✘ `door_close__903eb9s14` CLAP 0.276, PQ 6.33, RMS 0.130 — AST: слышно Explosion 0.28, а Door только 0.13. AST: Explosion 0.28, Door 0.13, Slam 0.12
- ✘ `door_close__903eb9s13` CLAP 0.236, PQ 6.06, RMS 0.075 — AST: слышно Knock 0.88, а Door только 0.42. AST: Knock 0.88, Door 0.42, Thunk 0.07
- ✘ `door_close__903eb9s17` CLAP 0.229, PQ 6.12, RMS 0.228 — AST: слышно Knock 0.25, а Door только 0.06. AST: Knock 0.25, Slam 0.13, Sound effect 0.06
- ✘ `door_close__903eb9s11` CLAP 0.198, PQ 6.64, RMS 0.116 — AST: слышно Knock 0.70, а Door только 0.18. AST: Knock 0.70, Door 0.18, Thunk 0.15
- ✘ `door_close__903eb9s15` CLAP 0.196, PQ 6.50, RMS 0.048 — AST: слышно Knock 0.84, а Door только 0.18. AST: Knock 0.84, Door 0.18, Thunk 0.07

## floor_walk — цель: heavy boots walking on wooden floorboards; ждём: Walk, footsteps, Creak
- ✔ `floor_walk__9e5a6ds11` CLAP 0.136, PQ 7.39, RMS 0.083 — AST: слышно Zipper (clothing) 0.11, а Walk, footsteps только 0.00. AST: Sound effect 0.53, Zipper (clothing) 0.11, Scrape 0.09
- ✘ `floor_walk__9e5a6ds12` CLAP 0.135, PQ 6.22, RMS 0.025 — AST: слышно Music 0.06, а Creak только 0.01. AST: Sound effect 0.14, Music 0.06, Knock 0.03
- ✘ `floor_walk__9e5a6ds13` CLAP 0.116, PQ 5.37, RMS 0.058 — AST: слышно Knock 0.14, а Walk, footsteps только 0.00. AST: Knock 0.14, Sound effect 0.11, Door 0.01
- ✘ `floor_walk__9e5a6ds14` CLAP 0.089, PQ 6.41, RMS 0.089 — AST: слышно Roar 0.48, а Walk, footsteps только 0.00. AST: Roar 0.48, Sound effect 0.42, Grunt 0.05
- ✘ `floor_walk__9e5a6ds17` CLAP 0.026, PQ 6.30, RMS 0.094 — AST: слышно Scrape 0.05, а Walk, footsteps только 0.00; CLAP: «a car engine revving» 1.00. AST: Sound effect 0.52, Scrape 0.05, Sigh 0.03
- ✘ `floor_walk__9e5a6ds15` CLAP 0.026, PQ 6.17, RMS 0.151 — AST: слышно Roar 0.12, а Creak только 0.00; CLAP: «a car engine revving» 1.00. AST: Sound effect 0.33, Roar 0.12, Grunt 0.10
- ✘ `floor_walk__9e5a6ds16` CLAP 0.001, PQ 6.28, RMS 0.168 — AST: слышно Roar 0.17, а Walk, footsteps только 0.00. AST: Sound effect 0.44, Roar 0.17, Grunt 0.11
- ✘ `floor_walk__9e5a6ds18` CLAP -0.023, PQ 7.04, RMS 0.162 — AST: слышно Roar 0.70, а Walk, footsteps только 0.00; CLAP: «a car engine revving» 0.99. AST: Roar 0.70, Sound effect 0.29, Roaring cats (lions, tigers) 0.05

## table_thud — цель: heavy object thudding on a table; ждём: Thump, thud, Thunk, Knock
- ✔ `table_thud__9565fas11` CLAP 0.163, PQ 7.05, RMS 0.056 — годен. AST: Knock 0.42, Sound effect 0.08, Thunk 0.05
- ✘ `table_thud__9565fas13` CLAP 0.158, PQ 7.25, RMS 0.057 — AST: слышно Wood 0.12, а Thunk только 0.00. AST: Wood 0.12, Hammer 0.04, Sound effect 0.03
- ✘ `table_thud__9565fas12` CLAP 0.097, PQ 7.48, RMS 0.015 — AST: слышно Dishes, pots, and pans 0.17, а Thump, thud только 0.00. AST: Dishes, pots, and pans 0.17, Cutlery, silverware 0.08, Chink, clink 0.07
- ✘ `table_thud__9565fas14` CLAP 0.008, PQ 7.68, RMS 0.050 — AST: слышно Breaking 0.08, а Knock только 0.00. AST: Breaking 0.08, Dishes, pots, and pans 0.05, Crack 0.05

## cough — цель: deep wet rasping cough; ждём: Cough, Throat clearing
- ✘ `cough__3a7ac3s12` CLAP 0.200, PQ 5.82, RMS 0.117 — AST: слышно Stomach rumble 0.89, а Throat clearing только 0.00. AST: Stomach rumble 0.89, Sound effect 0.07, Speech 0.02
- ✘ `cough__3a7ac3s16` CLAP 0.153, PQ 5.49, RMS 0.118 — AST: слышно Grunt 0.19, а Throat clearing только 0.00. AST: Sound effect 0.30, Grunt 0.19, Roar 0.09
- ✘ `cough__3a7ac3s13` CLAP 0.128, PQ 6.15, RMS 0.325 — AST: слышно Roar 0.09, а Throat clearing только 0.00; CLAP: «a car engine revving» 0.99. AST: Roar 0.09, Grunt 0.05, Sound effect 0.05
- ✘ `cough__3a7ac3s14` CLAP 0.105, PQ 6.17, RMS 0.201 — AST: слышно Sigh 0.14, а Throat clearing только 0.00. AST: Sound effect 0.15, Sigh 0.14, Explosion 0.05
- ✘ `cough__3a7ac3s15` CLAP 0.112, PQ 4.18, RMS 0.184 — AST: слышно Vehicle 0.08, а Throat clearing только 0.00. AST: Vehicle 0.08, Engine 0.07, Accelerating, revving, vroom 0.05
- ✘ `cough__3a7ac3s18` CLAP 0.053, PQ 5.42, RMS 0.416 — AST: слышно Sigh 0.11, а Throat clearing только 0.01; CLAP: «a car engine revving» 0.82. AST: Sound effect 0.12, Sigh 0.11, Gasp 0.07
- ✔ `cough__3a7ac3s11` CLAP 0.035, PQ 5.72, RMS 0.126 — AST: слышно Grunt 0.08, а Throat clearing только 0.00. AST: Sound effect 0.25, Grunt 0.08, Sigh 0.08
- ✘ `cough__3a7ac3s17` CLAP -0.073, PQ 4.07, RMS 0.323 — AST: слышно Snort 0.08, а Throat clearing только 0.00; CLAP: «distorted white noise» 0.95. AST: Snort 0.08, Sound effect 0.06, Scrape 0.05

## fire_catching — цель: dry wood fire catching and growing; ждём: Fire, Crackle
- ✘ `fire_catching__4a92b0s16` CLAP 0.509, PQ 5.39, RMS 0.220 — AST: слышно Throbbing 0.14, а Fire только 0.06. AST: Throbbing 0.14, Fire 0.06, Hum 0.05
- ✔ `fire_catching__4a92b0s18` CLAP 0.433, PQ 6.01, RMS 0.073 — годен. AST: Sizzle 0.32, Fire 0.21, Frying (food) 0.13
- · `fire_catching__4a92b0s15` CLAP 0.415, PQ 5.50, RMS 0.086 — годен. AST: Fire 0.16, Rumble 0.11, Vehicle 0.05
- · `fire_catching__4a92b0s13` CLAP 0.398, PQ 5.84, RMS 0.029 — годен. AST: Fire 0.18, Noise 0.05, Finger snapping 0.02
- ✘ `fire_catching__4a92b0s12` CLAP 0.293, PQ 6.21, RMS 0.005 — AST: слышно White noise 0.11, а Fire только 0.01. AST: White noise 0.11, Hiss 0.09, Steam 0.08
- ✘ `fire_catching__4a92b0s14` CLAP 0.295, PQ 6.08, RMS 0.005 — AST: слышно Hum 0.08, а Fire только 0.02. AST: Silence 0.10, Hum 0.08, Static 0.07
- ✘ `fire_catching__4a92b0s17` CLAP 0.256, PQ 7.06, RMS 0.000 — AST: слышно Tick 0.48, а Fire только 0.00; тихо: RMS 0.0004. AST: Tick 0.48, Tick-tock 0.17, Clicking 0.09
- ✘ `fire_catching__4a92b0s11` CLAP 0.247, PQ 6.04, RMS 0.071 — AST: слышно Steam 0.13, а Fire только 0.01. AST: Steam 0.13, Hiss 0.12, White noise 0.08
