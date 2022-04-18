import datetime



tt = '20220415'
assert len(tt) == 8
assert datetime.datetime.strptime(tt, '%Y%m%d')
