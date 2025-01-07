import datetime

# 导入timezone类
from datetime import timezone

# 获取当前日期和时间
date_with_timezone = datetime.datetime.now(timezone.utc)

# 创建带有时区信息的日期和时间
# date_with_timezone = current_date.replace(tzinfo=timezone.utc)

print("带有时区信息的日期和时间:", date_with_timezone)