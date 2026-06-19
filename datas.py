import requests
from bs4 import BeautifulSoup
import time
import os
import random
import logging
from urllib.parse import urljoin

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraper.log'
)
logger = logging.getLogger(__name__)

# 随机User-Agent池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15"
]

# 省份配置
provinces = [
    {"name": "北京", "url": "https://you.ctrip.com/sight/beijing1", "pages": 100},
    {"name": "天津", "url": "https://you.ctrip.com/sight/tianjin154", "pages": 100},
    {"name": "河北", "url": "https://you.ctrip.com/sight/hebei100059", "pages": 100},
    {"name": "山西", "url": "https://you.ctrip.com/sight/shanxi100056", "pages": 100},
    {"name": "内蒙古", "url": "https://you.ctrip.com/sight/innermongolia100062", "pages": 100},
    {"name": "辽宁", "url": "https://you.ctrip.com/sight/liaoning100061", "pages": 100},
    {"name": "吉林", "url": "https://you.ctrip.com/sight/jilin267", "pages": 100},
    {"name": "黑龙江", "url": "https://you.ctrip.com/sight/heilongjiang100055", "pages": 100},
    {"name": "上海", "url": "https://you.ctrip.com/sight/shanghai2", "pages": 100},
    {"name": "江苏", "url": "https://you.ctrip.com/sight/jiangsu100066", "pages": 100},
    {"name": "浙江", "url": "https://you.ctrip.com/sight/zhejiang100065", "pages": 100},
    {"name": "安徽", "url": "https://you.ctrip.com/sight/anhui100068", "pages": 100},
    {"name": "福建", "url": "https://you.ctrip.com/sight/fujian100038", "pages": 100},
    {"name": "江西", "url": "https://you.ctrip.com/sight/jiangxi100054", "pages": 100},
    {"name": "山东", "url": "https://you.ctrip.com/sight/shandong100039", "pages": 100},
    {"name": "河南", "url": "https://you.ctrip.com/sight/henan100058", "pages": 100},
    {"name": "湖北", "url": "https://you.ctrip.com/sight/hubei100067", "pages": 100},
    {"name": "湖南", "url": "https://you.ctrip.com/sight/hunan100053", "pages": 100},
    {"name": "广东", "url": "https://you.ctrip.com/sight/guangdong100051", "pages": 100},
    {"name": "广西", "url": "https://you.ctrip.com/sight/guangxi100052", "pages": 100},
    {"name": "海南", "url": "https://you.ctrip.com/sight/hainan100001", "pages": 100},
    {"name": "重庆", "url": "https://you.ctrip.com/sight/chongqing158", "pages": 100},
    {"name": "四川", "url": "https://you.ctrip.com/sight/sichuan100009", "pages": 100},
    {"name": "贵州", "url": "https://you.ctrip.com/sight/guizhou100064", "pages": 100},
    {"name": "云南", "url": "https://you.ctrip.com/sight/yunnan100007", "pages": 100},
    {"name": "西藏", "url": "https://you.ctrip.com/sight/tibet100003", "pages": 100},
    {"name": "陕西", "url": "https://you.ctrip.com/sight/shaanxi100057", "pages": 100},
    {"name": "甘肃", "url": "https://you.ctrip.com/sight/gansu100060", "pages": 100},
    {"name": "青海", "url": "https://you.ctrip.com/place/qinghai100032", "pages": 100},
    {"name": "宁夏", "url": "https://you.ctrip.com/place/ningxia100063", "pages": 100},
    {"name": "新疆", "url": "https://you.ctrip.com/sight/xinjiang100008", "pages": 100},
    {"name": "香港", "url": "https://you.ctrip.com/sight/hongkong38", "pages": 100},
    {"name": "澳门", "url": "https://you.ctrip.com/sight/macau39", "pages": 100},
    {"name": "台湾", "url": "https://you.ctrip.com/sight/taiwan100076", "pages": 100}
]
# provinces = [
#     {"name": "青海", "url": "https://you.ctrip.com/sight/qinghai100032", "pages": 100},
#     {"name": "宁夏", "url": "https://you.ctrip.com/sight/ningxia100063", "pages": 100},
#
# ]



# 创建数据目录
os.makedirs("data", exist_ok=True)


def get_random_headers():
    """生成随机请求头"""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://you.ctrip.com/",
        "Connection": "keep-alive"
    }


def get_page(url, max_retries=3):
    """获取网页内容，支持重试机制"""
    for attempt in range(max_retries):
        try:
            headers = get_random_headers()
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            response.encoding = "utf-8"
            return response.text
        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {url}, 尝试 {attempt + 1}/{max_retries}, 错误: {e}")
            if attempt < max_retries - 1:
                # 指数退避重试
                wait_time = 2 ** attempt + random.random()
                logger.info(f"等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
    return None


def parse_attraction_item(item):
    """解析单个景点信息"""
    try:
        # 提取景点名称
        name_elem = item.find("div", class_="titleModule_name__Li4Tv")
        name = name_elem.find('span').find('a').get_text(strip=True) if name_elem else "#"

        # 提取景点级别
        level_elem = item.find('span', class_='titleModule_level-text-view__40Dbg')
        level = level_elem.get_text(strip=True) if level_elem else "#"

        # 提取热度
        hot_elem = item.find("span", class_="commentInfoModule_heat-score_value__J8p3b")
        hot = hot_elem.get_text(strip=True) if hot_elem else "#"

        # 提取口碑
        label_elem = item.find("span", class_="rankInfoModule_rank_desc_text__QY4cm")
        label = label_elem.get_text(strip=True) if label_elem else "#"

        # 提取评论数
        comment_elem = item.find_all('span', class_='commentInfoModule_comment-text__UBk1F')
        comment = comment_elem[2].get_text(strip=True) if len(comment_elem) > 2 else "#"

        # 提取标签
        title_elem = item.find('div', class_='rankInfoModule_tag_list_view__4_nZC')
        title = "#"
        if title_elem:
            elem_list = title_elem.find_all('span', class_='rankInfoModule_tag_text__FCSHe')
            if elem_list:
                title = ','.join([i.get_text(strip=True) for i in elem_list])

        # 提取地点
        location_elem = item.find("div", class_="distanceView_box__zWu29")
        location = location_elem.get_text(strip=True) if location_elem else "#"

        return {
            "景点名称": name,
            "地点": location,
            "景点级别": level,
            "热度": hot,
            "口碑": label,
            "标签": title,
            "评论": comment,
        }
    except Exception as e:
        logger.error(f"解析景点信息失败: {e}")
        return None


def scrape_province(province):
    """爬取单个省份的景点数据"""
    province_name = province["name"]
    base_url = province["url"]
    total_pages = province["pages"]

    logger.info(f"开始爬取 {province_name} 的景点数据")
    print(f"开始爬取 {province_name} 的景点数据...")

    attractions = []

    # 确定URL分页格式（不同省份可能有差异）
    if "place" in base_url:
        # 青海、宁夏等使用 /place/ 路径的省份
        url_template = f"{base_url}/s0-p{{}}.html"
    else:
        # 其他省份使用 /sight/ 路径
        url_template = f"{base_url}/s0-p{{}}.html"

    for page in range(1, total_pages + 1):
        current_url = url_template.format(page)
        print(f"正在爬取 {province_name} 第 {page} 页...")
        logger.info(f"爬取 {province_name} 第 {page} 页: {current_url}")

        # 随机延迟，避免被反爬
        time.sleep(random.uniform(2, 5))

        html_content = get_page(current_url)
        if not html_content:
            logger.error(f"获取 {province_name} 第 {page} 页失败，跳过")
            continue

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            content = soup.find("div", class_="cardListBox_box__lMuWz")

            if not content:
                logger.warning(f"第 {page} 页未找到景点列表容器，可能已到最后一页或页面结构变化")
                break

            item_list = content.find_all("div", class_="sightItemCard_box__2FUEj")

            if not item_list:
                logger.warning(f"第 {page} 页无景点数据，可能已到最后一页")
                break

            for item in item_list:
                attraction = parse_attraction_item(item)
                if attraction:
                    attractions.append(attraction)

                    # 每条数据解析后随机延迟
                    time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            logger.error(f"解析 {province_name} 第 {page} 页失败: {e}")
            continue

    # 保存数据
    if attractions:
        save_data(attractions, province_name)
        logger.info(f"成功爬取 {province_name} {len(attractions)} 条景点数据")
        print(f"成功爬取 {province_name} {len(attractions)} 条景点数据")
    else:
        logger.warning(f"未爬取到 {province_name} 的景点数据")
        print(f"未爬取到 {province_name} 的景点数据")

    return attractions


def save_data(attractions, province_name):
    """保存景点数据到文件"""
    file_path = f"data/{province_name}.txt"

    txt_content = "景点名称;地点;景点级别;热度;口碑;标签;评论"
    txt_content += "\n----------------------------------------\n"

    for attr in attractions:
        txt_content += f"{attr['景点名称']};{attr['地点']};{attr['景点级别']};{attr['热度']};{attr['口碑']};{attr['标签']};{attr['评论']}\n"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(txt_content)

    logger.info(f"数据已保存到: {file_path}")


def main():
    """主函数：爬取所有省份的景点数据"""
    all_attractions = []
    total_provinces = len(provinces)

    print(f"开始爬取全国景点数据，共 {total_provinces} 个省份...")
    logger.info(f"开始爬取全国景点数据，共 {total_provinces} 个省份")

    for i, province in enumerate(provinces, 1):
        print(f"\n===== [{i}/{total_provinces}] 爬取 {province['name']} =====")

        try:
            attractions = scrape_province(province)
            all_attractions.extend(attractions)

            # 省份之间随机延迟，避免被反爬
            if i < total_provinces:
                wait_time = random.uniform(5, 10)
                print(f"等待 {wait_time:.2f} 秒后继续爬取下一个省份...")
                logger.info(f"等待 {wait_time:.2f} 秒后继续爬取")
                time.sleep(wait_time)

        except Exception as e:
            logger.error(f"爬取 {province['name']} 时发生错误: {e}")
            print(f"爬取 {province['name']} 时发生错误: {e}")

    # 保存所有数据到一个文件
    if all_attractions:
        save_data(all_attractions, "全国景点")
        print(f"\n全部爬取完成！共获取 {len(all_attractions)} 条景点数据")
        logger.info(f"全部爬取完成！共获取 {len(all_attractions)} 条景点数据")
    else:
        print("\n未获取到任何景点数据！")
        logger.warning("未获取到任何景点数据！")


if __name__ == "__main__":
    main()

# import requests
# from bs4 import BeautifulSoup
# import time
#
# # 请求头（模拟浏览器访问）
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
#     "Accept-Language": "zh-CN,zh;q=0.9"
# }
#
# # 基础URL（注意：需确认分页参数是否为`page`，若不同需调整）
# base_url = "https://you.ctrip.com/sight/yunnan100007"
# total_pages = 100 # 爬取页数
# attractions = []  # 存储所有数据
#
# for page in range(1, total_pages + 1):
#     # 构造当前页URL（示例：?page=1, ?page=2）
#     current_url = f"{base_url}/s0-p{page}.html"
#
#     try:
#         print(f"正在爬取第 {page} 页...")
#         time.sleep(3)  # 页面请求间隔（防反爬）
#         response = requests.get(current_url, headers=headers, timeout=10)
#         response.raise_for_status()
#         response.encoding = "utf-8"  # 处理中文编码
#
#         soup = BeautifulSoup(response.text, "html.parser")
#         content = soup.find("div", class_="cardListBox_box__lMuWz")
#         item_list = content.find_all("div", class_="sightItemCard_box__2FUEj")
#
#         if not item_list:
#             print(f"第 {page} 页无数据，可能已到最后一页")
#             break  # 提前终止循环
#
#         for item in item_list:
#             name = item.find("div", class_="titleModule_name__Li4Tv").find('span').find('a').get_text(strip=True)
#
#             label_elem = item.find("span", class_="rankInfoModule_rank_desc_text__QY4cm")
#             label = label_elem.get_text(strip=True) if label_elem else "#"
#
#             hot_elem = item.find("span", class_="commentInfoModule_heat-score_value__J8p3b")
#             hot = hot_elem.get_text(strip=True) if hot_elem else "#"
#
#             score_elem = item.find("span",
#                                    class_="commentInfoModule_comment-text__UBk1F commentInfoModule_comment-score_value__iUsa8")
#             score = score_elem.get_text(strip=True) if score_elem else "#"
#
#             comment_elem = item.find_all('span', class_='commentInfoModule_comment-text__UBk1F')
#             comment = comment_elem[2].get_text(strip=True) if comment_elem else "#"
#
#             title_elem = item.find('div', class_='rankInfoModule_tag_list_view__4_nZC')
#
#             if title_elem:
#                 # 找到所有符合条件的 span 标签（返回 ResultSet 对象）
#                 elem_list = title_elem.find_all('span', class_='rankInfoModule_tag_text__FCSHe')
#
#                 # 修正：对列表中的每个元素调用 get_text()
#                 # 错误写法：elem = [elem_list.get_text(strip=True) for i in elem_list]
#                 # 正确写法：对每个元素 i 调用 get_text()
#                 elem = [i.get_text(strip=True) for i in elem_list]
#
#                 # 用逗号连接所有文本
#                 title = ','.join(elem)
#             else:
#                 title = "#"
#
#             level_elem = item.find('span', class_='titleModule_level-text-view__40Dbg')
#             level = level_elem.get_text(strip=True) if level_elem else "#"
#
#             location = item.find("div", class_="distanceView_box__zWu29").get_text(strip=True)
#
#             attractions.append({
#                 "景点名称": name,
#                 "地点": location,
#                 "景点级别": level,
#                 "热度": hot,
#                 "口碑": label,
#                 "标签": title,
#                 "评论": comment,
#             })
#
#             # 每条数据解析后添加短暂等待（可选）
#             time.sleep(1)  # 等待 0.5 秒（根据网站反爬强度调整）
#
#     except requests.exceptions.RequestException as e:
#         print(f"第 {page} 页请求失败: {e}")
#         continue  # 跳过当前页，继续下一页
#     except Exception as e:
#         print(f"第 {page} 页解析失败: {e}")
#         continue  # 跳过当前页，继续下一页
#
# txt_content = "景点名称;地点;景点级别;热度;口碑;标签;评论"
# txt_content += "\n----------------------------------------\n"
# for attr in attractions:
#     # 使用制表符对齐（可能需要根据实际内容调整）
#     txt_content += f"{attr['景点名称']};{attr['地点']};{attr['景点级别']};{attr['热度']};{attr['口碑']};{attr['标签']};{attr['评论']}\n"
#
# # 保存到 TXT 文件
# with open("./data/", "w", encoding="utf-8") as f:
#     f.write(txt_content)
#
# print(f"成功爬取 {len(attractions)} 条数据，保存至")