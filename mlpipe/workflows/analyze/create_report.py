from mlpipe.config.app_settings import get_reporting_config
from bs4 import BeautifulSoup

from mlpipe.utils.file_tool import read_text_file_lines


def generate_html_reports(json_str: str):
    config = get_reporting_config()
    html_string = "".join(read_text_file_lines(config['html_template_path']))
    soup = BeautifulSoup(html_string, "html.parser")

    # setup script remote links
    for tag in soup.find_all(lambda t: t.name == "script" and not t.attrs['src'].startswith("http")):
        tag.attrs['src'] = config['hosting_url_prefix'] + tag.attrs['src']

    # setup style remote links
    for tag in soup.find_all(lambda t: t.name == "link" and not t.attrs['href'].startswith("http")):
        tag.attrs['href'] = config['hosting_url_prefix'] + tag.attrs['href']

    # setup base href
    soup.find('base').attrs['href'] = config['base_href']

    # embed data
    script_tag = soup.new_tag(name="script", type="text/javascript")
    script_tag.append("window.sensor_data = " + json_str + ";")
    soup.find('head').append(script_tag)

    return soup.prettify()


def generate_html_report(json_str: str, output_path: str):
    html = generate_html_reports(json_str)
    with open(output_path, "w") as f:
        f.write(html)

