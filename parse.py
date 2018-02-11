import csv
import math


def signed_log(num):
    def sign(n):
        if n == 0:
            return 0
        return -1 if n < 0 else 1
    return 0 if abs(num) <= 1 else sign(num) * math.log(abs(num))


def parse_numbers_in_list(l, log_transform=False):
    def normalize_number(num):
        if type(num) is str:
            return num.replace(',', '').replace('$', '').replace('%', '')
        return num
    results = [float(normalize_number(e)) if e else None for e in l]
    if log_transform:
        return [signed_log(e) if e is not None else None for e in results]
    return results


def parse_crv_data(filename='crv-crypto-data.csv'):
    x_mat = []
    y_mat = []  # Return a list of possible objectives rather than a single
                # objective
    feature_labels = ['Total # Commits', 'Total # Contributors',
                      'One Month # Commits', 'One Month # Contributors',
                      'Telegram Members in Top Group', 'Reddit Members',
                      'Exchanges Listed', '# of Top 5 Exchanges',
                      '# of Hashtag Tweets (30 days)',
                      '# of News Mentions (30 days)', 'Twitter Followers']
    objective_labels = ['One Month Return', 'One Year Return', 'Market Cap']
    with open(filename, 'r') as f:
        csvreader = csv.reader(f)
        next(csvreader, None)  # Skip header
        for line in csvreader:
            project, commits, contributors, month_commits, month_contributors,\
            _, _, _, telegram, reddit, exchanges, top_exchanges, month_return,\
            year_return, market_cap, _, _, month_twitter_mentions,\
            month_news_mentions, _, _, twitter = line
            x_mat.append(parse_numbers_in_list([
                commits, contributors, month_commits, month_contributors,
                telegram, reddit, exchanges, top_exchanges,
                month_twitter_mentions, month_news_mentions, twitter
            ], log_transform=True))
            y_mat.append(parse_numbers_in_list(
                [month_return, year_return, market_cap], log_transform=True
            ))
    return (x_mat, y_mat, feature_labels, objective_labels)
