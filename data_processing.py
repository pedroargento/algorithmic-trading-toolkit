import os
from typing import NamedTuple, Generator, Callable
from datetime import date, datetime
from itertools import chain

class Ticker(NamedTuple):
    date: datetime
    code: str
    type_of_market: str
    price_open: float
    price_max: float
    price_min: float
    price_med: float
    price_last: float
    best_bid: float
    best_ask: float
    trading_volume: float
    correction_factor: int

def process_b3_line(line: str) -> Ticker:
    return Ticker(date = datetime.strptime(line[2:10], '%Y%m%d'),
                 code = line[12:24].rstrip(),
                 type_of_market = line[24:27],
                 price_open = float(line[56:69])/100,
                 price_max = float(line[69:82])/100,
                 price_min = float(line[82:95])/100,
                 price_med = float(line[95:108])/100,
                 price_last = float(line[108:121])/100,
                 best_bid = float(line[121:134])/100,
                 best_ask = float(line[134:147])/100,
                 trading_volume = float(line[152:170])/100,
                 correction_factor = int(line[210:217])
            ) 

def process_b3_file(file_path: str) -> Generator[Ticker, None, None]:
    for line in open(file_path, 'r'):
        if(line[0:2] == '01'):
            yield process_b3_line(line)

def process_folder(path: str, processor: Callable) -> Generator[Ticker, None, None]:
    files = (os.path.join(path, f) for f in os.listdir(path))
    for file_generator in files:
        for ticker in processor(file_generator):
            yield ticker

        
