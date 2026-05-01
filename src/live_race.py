import fastf1
import datetime
import json
import os

fastf1.Cache.enable_cache('data')


def get_race_calendar(year=2026, n_upcoming=3):
    """
    Get the next n upcoming F1 race weekends.
    Returns list of dicts with race info and countdown.
    """
    try:
        schedule = fastf1.get_event_schedule(year)
        now = datetime.datetime.now(datetime.timezone.utc)

        upcoming = []
        for _, event in schedule.iterrows():
            try:
                race_date = event['EventDate']
                if hasattr(race_date, 'tzinfo') and race_date.tzinfo is None:
                    race_date = race_date.replace(tzinfo=datetime.timezone.utc)

                if race_date > now:
                    delta = race_date - now
                    days = delta.days
                    hours = delta.seconds // 3600

                    upcoming.append({
                        'round': int(event['RoundNumber']),
                        'name': event['EventName'],
                        'location': event['Location'],
                        'country': event['Country'],
                        'date': race_date.strftime('%Y-%m-%d'),
                        'days_away': days,
                        'hours_away': hours,
                        'is_this_weekend': days <= 3,
                        'is_today': days == 0,
                    })

                    if len(upcoming) >= n_upcoming:
                        break

            except Exception:
                continue

        return upcoming

    except Exception as e:
        print(f"Calendar fetch failed: {e}")
        return []


def is_race_weekend():
    """Check if there is an F1 race happening this weekend."""
    upcoming = get_race_calendar(n_upcoming=1)
    if upcoming and upcoming[0]['days_away'] <= 3:
        return True, upcoming[0]
    return False, None


def get_live_session_info():
    """
    Check if there is a live F1 session right now.
    Returns session info if live, None otherwise.
    """
    is_weekend, race_info = is_race_weekend()
    if not is_weekend or race_info is None:
        return None

    try:
        year = datetime.datetime.now().year
        session = fastf1.get_session(year, race_info['round'], 'R')
        return {
            'race': race_info['name'],
            'round': race_info['round'],
            'is_live': True,
            'session': session
        }
    except Exception:
        return None


def get_calendar_for_api(n=3):
    """Return calendar data formatted for the dashboard API."""
    upcoming = get_race_calendar(n_upcoming=n)
    result = []
    for race in upcoming:
        result.append({
            'name': race['name'],
            'location': race['location'],
            'date': race['date'],
            'days_away': race['days_away'],
            'is_this_weekend': race['is_this_weekend'],
            'badge': '🔴 LIVE' if race['is_today'] else
                     '🟡 THIS WEEKEND' if race['is_this_weekend'] else
                     f"📅 {race['days_away']} days"
        })
    return result


if __name__ == '__main__':
    print("Fetching F1 calendar...")
    races = get_calendar_for_api(n=5)
    for race in races:
        print(f"{race['badge']:25} {race['name']} — {race['date']}")

    is_weekend, info = is_race_weekend()
    print(f"\nRace this weekend: {is_weekend}")
    if info:
        print(f"  {info['name']} in {info['location']}") 