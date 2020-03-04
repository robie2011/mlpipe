from bs4 import BeautifulSoup

from mlpipe.config.app_settings import AppConfig
from mlpipe.utils.file_tool import read_text_file_lines, write_text_file
from mlpipe.utils.path_tool import get_abspath_or_relpath


def generate_html_reports(json_str: str):
    tpl_path = get_abspath_or_relpath(AppConfig["reporting.html_template_path"])
    html_string = "".join(read_text_file_lines(tpl_path))

    soup = BeautifulSoup(html_string, "html.parser")

    url_prefix = AppConfig["reporting.hosting_url_prefix"]
    # setup script remote links
    for tag in soup.find_all(lambda t: t.name == "script" and not t.attrs['src'].startswith("http")):
        tag.attrs['src'] = url_prefix + tag.attrs['src']

    # setup style remote links
    for tag in soup.find_all(lambda t: t.name == "link" and not t.attrs['href'].startswith("http")):
        tag.attrs['href'] = url_prefix + tag.attrs['href']

    # setup base href
    soup.find('base').attrs['href'] = AppConfig["reporting.base_href"]

    # embed data
    script_tag = soup.new_tag(name="script", type="text/javascript")
    script_tag.append("window.sensor_data = " + json_str + ";")
    soup.find('head').append(script_tag)

    return soup.prettify()


def generate_html_report(json_str: str, output_path: str):
    html = generate_html_reports(json_str)
    write_text_file(path=output_path, text=html)
