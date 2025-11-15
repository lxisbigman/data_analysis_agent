from jinja2 import Template
from typing import Dict

def load_templates() -> Dict[str, Template]:
    templates = {
        "overview": Template("""
1、综合画像

1）A占比前三办事处：{{ top_a_offices|join('、') }}A占比均为100%；

2）CD占比后三办事处：{{ bottom_cd_offices|join('、') }}；

3）TOP优秀网络点评
{% for network in top_networks %}
①{{ network.name }}：{{ network.network_count }}张网络A占比{{ network.a_ratio }}, {{ network.dimensions_completed }}, 网络运行稳定，客户满意度高；
{% endfor %}
"""),
        "cd_analysis": Template("""
4）CD级网络分析

①网元质量：共涉及{{ cd_analysis.net_element.count }}个网络，{{ cd_analysis.net_element.suggestion }}。

②运维质量：共涉及{{ cd_analysis.operation.count }}个网络，{{ cd_analysis.operation.suggestion }}，涉及的网络见附件清单。

③网络安全：共涉及{{ cd_analysis.security.count }}个网络，{{ cd_analysis.security.suggestion }}，涉及的网络见附件清单。

④网络负荷：共涉及{{ cd_analysis.load.count }}个，{{ cd_analysis.load.suggestion }}。

⑤参数规范：共涉及{{ cd_analysis.parameters.count }}个网络，{{ cd_analysis.parameters.suggestion }}，涉及的网络见附件清单。

⑥用户感知：共涉及{{ cd_analysis.perception.count }}张网络全A，{{ cd_analysis.perception.suggestion }}。
"""),
        "hardware_fix": Template("""
5）硬件问题整改

①9000E隐患单板整改：

截止24年Q2月共识别{{ hardware.faulty_9000e.total }}块隐患单板，当前完成{{ hardware.faulty_9000e.fixed }}处隐患处理。请{{ hardware.faulty_9000e.provinces|join('、') }}等省份尽快完成隐患单板治理，并上传平台更新。

②61V5结温治理：

目前共识别隐患单板{{ hardware.temperature.total }}块，10-11月新增隐患单板{{ hardware.temperature.new_added }}块，新增整改单板{{ hardware.temperature.new_fixed }}块，共整改隐患单板{{ hardware.temperature.fixed }}块，总体整改率{{ hardware.temperature.rate }}%。当前无超期未整改隐患单板，请相关办事处尽快推动替换。

③300W硫化整改：

9000E 300W 24年共识别必整改硫化单板{{ hardware.sulfuration.total }}块，当前已提交补发货申请{{ hardware.sulfuration.applied }}块，已到货{{ hardware.sulfuration.arrived }}, 已整改{{ hardware.sulfuration.fixed }}块，已回退{{ hardware.sulfuration.returned }}块，到货整改率{{ hardware.sulfuration.rate }}%。{{ hardware.sulfuration.provinces|join('、') }}需推进按计划完成整改逆向流程。
"""),
        "dimension_analysis": Template("""
2、网元质量维度：

A等级TOP3办事处为：{{ dimensions.net_element.top_a_offices|join('、') }}；C/D等级LAST3办事处为：{{ dimensions.net_element.bottom_cd_offices|join('、') }}。

3、运维质量维度：

A等级TOP共涉及{{ dimensions.operation.top_a_offices|join('/') }}共{{ dimensions.operation.top_a_count }}个处，A占比均为100%；C/D等级LAST3办事处为：{{ dimensions.operation.bottom_cd_offices|join('、') }}。

4、网络安全维度：

A等级TOP共涉及{{ dimensions.security.top_a_offices|join('/') }}共{{ dimensions.security.top_a_count }}个处，A占比均为100%；C/D等级LAST3办事处为：{{ dimensions.security.bottom_cd_offices|join('、') }}。

5、网络负荷维度：

A等级TOP共涉及{{ dimensions.load.top_a_offices|join('/') }}共{{ dimensions.load.top_a_count }}个处，A占比均为100%；C/D等级LAST3办事处为：{{ dimensions.load.bottom_cd_offices|join('、') }}。

6、参数规范维度：

A等级TOP3办事处为：{{ dimensions.parameters.top_a_offices|join('、') }}；C/D等级LAST3办事处为：{{ dimensions.parameters.bottom_cd_offices|join('、') }}。

7、用户感知维度：

除{{ dimensions.perception.excluded_offices|join('、') }}处外，其它办事处网络均为A等级，需继续加强网络维护，提升客户满意度。C/D等级LAST3办事处为：{{ dimensions.perception.bottom_cd_offices|join('、') }}。
"""),

        "old_equipment_backnet":Template("""
6)老旧设备退网

共计梳理{{ old_equipment_all }}%台高风险设备，其中{{ old_equipment_static }}%台设备纳入高风险强制替换清单，剩余{{ old_equipment_no_static }}%台纳入次高风险建议替换清单。
{{ old_equipment_year }}%年将继续联合总监办、MKT、产研、办事处全力推动这{{ old_equipment_static }}%台风险设备替换/割接退网，目前已完成{{ old_equipment_finish }}%台设备替换/退网割接。
请各省尽快安排实施。"""
        ),
        "net_quality": Template("""
三、各维度详细画像分析

1、网元质量画像

①网元质量画像结果

1. 网元质量维度评估画像A占比{{ net_quality_a_ratio }}%，B占比{{ net_quality_b_ratio }}%，C占比{{ net_quality_c_ratio }}%，D占比{{ net_quality_d_ratio }}%。其中A占比较上月上升{{ net_quality_a_increase }}pp，CD占比较上月下降{{ net_quality_cd_decrease }}pp。十一及亚运重保封网结束，办事处组织版本升级&补丁加载，网元质量CD等级占比下降。
2. A等级TOP3办事处为：{{ net_quality_top_a_offices|join('、') }}；
3. C/D等级LAST3办事处为：{{ net_quality_bottom_cd_offices|join('、') }}（{{ net_quality_bottom_reason }}）。
""")
    }
    return templates
