# mahjong-bot ♡

training a bot to play riichi mahjong, built a small version of the game, trained a bot on it by having it play against copies of itself, play here >>> 
https://mahjong-bot-17p7.onrender.com 


## What's in here

game engine: the wall, turn order, which moves are legal, riichi, tsumo and ron, furiten, and draws. for scoring, the [`mahjong`](https://pypi.org/project/mahjong/) library is used 

The bot is trained with PPO (`MaskablePPO` from sb3-contrib, which lets it ignore illegal moves). There are also two simple bots to compare it against: one that plays randomly, and one that always discards whatever tile brings its hand closest to winning (heuristic bot).

The demo is a FastAPI server and one HTML page.

## The rules

one hand at a time, four players, standard 136 tiles. There are no calls (no chi, pon or kan), which shrinks the game a lot
Riichi and ippatsu are in
one dora and no ura-dora
Furiten is checked
If the wall runs out it's a draw and nobody pays anyone and riichi sticks get handed back
If two people can ron on the same tile, whoever is next in turn order gets it

## to run 

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest mahjong_rl/tests -q     
```

To play:

```bash
python -m uvicorn mahjong_rl.serving.api:app --port 8010
```


The server loads `mahjong_rl/serving/models/model.zip` when it starts. You can point it somewhere else with `MAHJONG_MODEL=/path/to/model.zip`. If it can't find a model it falls back to the heuristic bot, and `/health` will say so. Games live in memory, so restarting the server wipes them

The API has four routes: `GET /health`, `POST /game/new`, `POST /game/{id}/action`, and `POST /move`. That last one is stateless. You send it a player's view of the game plus the legal actions and it returns the bot's choice.

There's also a `Dockerfile` that uses CPU-only PyTorch. 
I haven't been able to build it on my machine

## Training it

```bash
python -m mahjong_rl.training.train --run-name shaped5 --timesteps 3000000 --shaping-coef 0.05
tensorboard --logdir runs
```

 `runs/<name>/`
 win-rate curve (`eval.csv`) 
Every 10 iterations the current policy plays 400 hands against three heuristic bots. An iteration here is 4,096 of the bot's own decisions, and one hand is roughly 15 of those, so the 3M-step runs were about 200,000 hands each

The bot sees a 458-number description of the game from its own seat: its hand, everyone's discards, the dora, riichi flags, how much wall is left, the winds, whether it's furiten, its current shanten, and what the shanten would be after each possible discard

The reward is the hand's points divided by 8000, so a mangan is about 1. there's a small bonus each time a discard moves the hand closer to tenpai (`--shaping-coef`, and 0 turns it off)

To compare checkpoints on a fixed set of hands:

```bash
python -m mahjong_rl.training.compare runs/shaped5/latest.zip heuristic --episodes 4000 --seed 222
```

## How it did

I tested the final checkpoint on 4,000 hands it had never seen, against three heuristic bots, rotating seats. Here it is next to what the heuristic bot itself scores in the same setup:

| | wins | deals into someone else | avg points per hand |
|---|---|---|---|
| heuristic bot | 16.7% ± 1.2 | 12.4% | -9 |
| trained bot | 15.2% ± 1.1 | 10.3% | +164 |

So it wins a little less often than the heuristic, but deals in less and comes out ahead on points. That points gap is only about two standard errors. The fair summary is that the bot plays about as well as the heuristic 

400 hand evals during training showed it a few points ahead, but that was noise...

The reward bonus mattered a lot. With none, the bot was still at 0% wins after a million steps. A bonus of 0.01 got it to around 11%, and 0.05 got it to around 15% (from an earlier 2,000-hand test)

## Where things are

```
mahjong_rl/
  engine/     tiles, state, actions, game (the turn loop), scoring, shanten, observation, env
  agents/     random, heuristic, ppo
  training/   train, self_play, callbacks, evaluate, compare
  serving/    api, model_loader, models/model.zip
  web/        index.html
  tests/
```
