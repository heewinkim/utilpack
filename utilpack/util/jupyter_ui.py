# -*- coding: utf-8 -*-
"""
===============================================
widget util module
===============================================

========== ====================================
========== ====================================
 Module     widget util module
 Date       2020-03-26
 Author     heewinkim
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*
    * 데이터 처리관련 유틸모음

===============================================
"""
from utilpack.core import PyImage
from .image_util import PyImageUtil
from IPython.display import display, clear_output,HTML,Javascript
import ipywidgets
import time

backButtonActivate = False


class ErrorDecorator:

    def __init__(self,initFunc=None):
        self.initFunc=initFunc

    def __call__(self,func):

        def wrapper(*args,**kwargs):
            try:
                result = func(*args,**kwargs)
                if result:
                    return result
            except Exception as e:
                print(e)
                print('에러가 발생했습니다!')
                print("3초 후 자동으로 재시작합니다.")
                print('restart .. 3 ', end=' ')
                time.sleep(1)
                print('2 ', end=' ')
                time.sleep(1)
                print('1 ', end=' ')
                time.sleep(1)
                clear_output()
                if self.initFunc:
                    self.initFunc()
        return wrapper


class ButtonController(object):

    def __init__(self,Buttons):
        self.Buttons=Buttons
        self.descriptions=[btn.description for btn in Buttons]

    def active(self,descriptions=None):
        if not descriptions:
            descriptions = self.descriptions
        for btn,desc in zip(self.Buttons,descriptions):
            btn.description = desc
            btn.disabled = False

    def inactive(self,descriptions=None):
        if not descriptions:
            descriptions = ['출력중']*len(self.Buttons)
        for btn, desc in zip(self.Buttons, descriptions):
            btn.description = desc
            btn.disabled = True


class PyUI(object):

    ErrorDecorator = ErrorDecorator
    ButtonController = ButtonController

    @staticmethod
    def jupyterSummary(title, contents=None, author='heewinkim', numbering=False):
        """
        주피터 요약 설명문을 출력합니다.

        :param title: 제목
        :param contents: list of string which is descriptions
        :param author: author name
        :param numbering: use <ol> tag if numbering is true , else use <ul>
        :return: None
        """
        if contents is None:
            contents = []
        summary = HTML("""
        <div style="border: 5px ridge #ffffff; padding:10px;">
            <h3>{0}</h3>
            <h5 align="right">{1}</h5>
            <hr/>
            <{3}>
                {2}
            </{3}>
        </div>
        """.format(title, author, '\n'.join(['<li>' + content + '</li>' for content in contents]),
                   'ol' if numbering else 'ul'))
        display(summary)

    @staticmethod
    def select(dataList, options, runFunc,initFunc=None,backFunc=None,width=None, description='실행',**funcKwargs):
        """
        dataList를 options로 출력한뒤 선택된 option에 해당하는 data로 runFunc을 실행시킵니다.

        :param dataList: 실제 runFunc에 들어갈 데이터의 리스트
        :param options: 화면상에 출력된 각 데이터의 설명 리스트, dataList과 1:1 매칭
        :param runFunc: 실행될 함수, runFunc(data,**funcKwargs)->any
        :param initFunc: init function, 파라미터는 지원하지 않습니다. init()->any, None일경우 표시하지 않음
        :param description: runButton에 들어갈 텍스트
        :param width: widget of widgetStructure, support px, % , eg. "10px" or "50%" , default "auto"
        :param funcKwargs: runFunc에 들어갈 키워드변수
        :return: None
        """
        list_widget = ipywidgets.Select(options=options, rows=5,layout=ipywidgets.Layout(width='100%'))
        runButton = ipywidgets.Button(description=description)
        initButton = ipywidgets.Button(description='초기화면')
        backButton = ipywidgets.Button(description='이전화면')

        if len(dataList)==0:
            print('생성된 데이터가 없습니다.')
            display(initButton)
            return

        def onClickRun(value):
            selected_data = dataList[list_widget.index]
            runFunc(selected_data,initFunc=initFunc,backFunc=backFunc, **funcKwargs)

        runButton.on_click(onClickRun)
        initButton.on_click(lambda v:initFunc())
        backButton.on_click(lambda v:backFunc())

        widgetStructure = [
            [list_widget],
            [runButton]
        ]
        if not backFunc:
            backButton.layout.visibility = 'hidden'
        if not initFunc:
            initButton.layout.visibility = 'hidden'

        widgetStructure[-1].extend([v for v in [backButton,initButton] if v.layout.visibility!='hidden'])
        PyUI.appLayout(widgetStructure,width=width)
        return ButtonController([v for v in [runButton,backButton,initButton] if v.layout.visibility!='hidden'])

    @staticmethod
    def select_image(dataList, imreadFunc, runFunc, options, initFunc=None, backFunc=None, autoRun=False, width=None,view_h=400, description='실행', **funcKwargs):
        """
        dataList를 options로 출력한뒤 선택된 option에 해당하는 data로 runFunc을 실행시킵니다.

        :param dataList: 실제 runFunc에 들어갈 데이터의 리스트
        :param imreadFunc: dataList의 각 원소(이미지데이터)를 읽어들이는 함수 반환값은 cv2 타입의 이미지이어야 한다.
        :param runFunc: dataList의 원소를 받아 실행할 함수 def runFunc(dataList[n],resultView,**kwargs) -> None
        :param options: dataList를 화면에 뿌릴 때 나타날 각 데이터의 내용들
        :param resultViews: runFunc에서 만약 결과를 출력할경우 사용될 ipywidgets.Image의 리스트,
        :param controlWidgets: resultViews의 결과를 컨트롤할 ipywidget 리스트
        :param autoRun: runButton의 필요없이 select 항목이 바뀔때마다 자동으로 runFunc실행할지에 대한 boolean값
        :param options: 화면상에 출력된 각 데이터의 설명 리스트, dataList과 1:1 매칭
        :param initFunc: init function, 파라미터는 지원하지 않습니다. init()->any, None일경우 표시하지 않음
        :param description: runButton에 들어갈 텍스트
        :param width: widget of widgetStructure, support px, % , eg. "10px" or "50%" , default "auto"
        :param funcKwargs: runFunc에 들어갈 키워드변수,resultViews,controlWidgets를 포함합니다.

        :return: None
        """

        list_widget = ipywidgets.Select(options=options, rows=5, layout=ipywidgets.Layout(width='100%'))
        runButton = ipywidgets.Button(description=description)
        initButton = ipywidgets.Button(description='초기화면')
        backButton = ipywidgets.Button(description='이전화면')

        previewWidget = ipywidgets.Image(
            value=PyImage.cv2bytes(PyImageUtil.resize_image(imreadFunc(dataList[0]), height=view_h)),
            format='png',
            height=view_h,
        )

        if len(dataList) == 0:
            print('생성된 데이터가 없습니다.')
            display(initButton)
            return

        def onClickRun(value):
            selected_data = dataList[list_widget.index]
            runFunc(selected_data,imreadFunc=imreadFunc, initFunc=initFunc, backFunc=backFunc, **funcKwargs)

        def onClickSelect(value):
            previewWidget.value = PyImage.cv2bytes(
                PyImageUtil.resize_image(imreadFunc(dataList[value.owner.index]), height=view_h))
            if autoRun:
                onClickRun(0)

        runButton.on_click(onClickRun)
        initButton.on_click(lambda v: initFunc())
        backButton.on_click(lambda v: backFunc())
        list_widget.observe(onClickSelect, ['value'])

        widgetStructure = [
            [previewWidget]
        ]
        if 'resultViews' in funcKwargs:
            for resultView in funcKwargs['resultViews']:
                resultView.height = view_h
            widgetStructure[-1].extend(funcKwargs['resultViews'])
        if 'controlWidgets' in funcKwargs:
            widgetStructure.append(funcKwargs['controlWidgets'])
        widgetStructure.extend([
            [list_widget],
            [runButton]
        ])
        if not backFunc:
            backButton.layout.visibility = 'hidden'
        if not initFunc:
            initButton.layout.visibility = 'hidden'

        widgetStructure[-1].extend([v for v in [backButton, initButton] if v.layout.visibility != 'hidden'])
        PyUI.appLayout(widgetStructure, width=width)
        return ButtonController([v for v in [runButton, backButton, initButton] if v.layout.visibility != 'hidden'])

    @staticmethod
    def back(initFunc=None,backFunc=None,clearOutput=True,**kwargs):
        """
        init,back 버튼 및 버튼 콜백 함수를 정의합니다.

        :param initFunc: initFunc(**kwargs['init'])
        :param backFunc: backFunc(**kwargs['init'])
        :param clearOutput: boolean
        :param kwargs: kwargs, consist of 'init','back',
        :return: PyUI.ButtonController(initButton,backButton), ButtonController object offered active,inactive methods
        """
        if clearOutput:
            clear_output()
        initKwargs = kwargs.get('init',{})
        backKwargs = kwargs.get('back',{})
        initButton = ipywidgets.Button(description="처음으로")
        backButon = ipywidgets.Button(description="이전화면")
        if initFunc:
            initButton.on_click(lambda v: initFunc(**initKwargs) if initFunc else print('required initFunc'))
        else:
            initButton.layout.visibility = 'hidden'
        if backFunc:
            backButon.on_click(lambda v: backFunc(**backKwargs) if backFunc else print('required backFunc'))
        else:
            backButon.layout.visibility = 'hidden'

        widgetStructure = [
            [backButon,initButton]
        ]
        PyUI.appLayout(widgetStructure)

        return ButtonController([backButon,initButton])

    @staticmethod
    def appLayout(widgetStructure, run=True, width=None, HboxLayout=None, VboxLayout=None,**kwargs):
        """
        :param widgetStructure: list of list(elements are widget)
        :param width: widget of widgetStructure, support px, % , eg. "10px" or "50%" , default "auto"
        :param HboxLayout: if offered,It apply to Horizontal Box Layout
        :param VboxLayout: if offered,It apply to Vertical Box Layout
        :param kwargs: if offered 'logo' which is logo image ilfepath, logo included using 'logo_size' (default=['auto','auto'])
        :return: widget object
        make appLayout Style object

        outer list means vertical structure and
        inner list means horizontal structure
        if boxLayout offered, all HBox,VBox using boxLayout

        eg. [[Button,Button],[IntSlider]]

        -------------------
        | Button | Button |
        -------------------
        |    IntSlider    |
        -------------------

        all widgets width height size is evenly sperate with [ section size / len(widgets) ]
        if you want make empty space, include empty list (vertical) or None (horizontal) instead of widget

        eg. [[Button,None,Button],[],[IntSlider]]

        ----------------------------
        | Button |        | Button |
        ----------------------------
        |                          |
        ----------------------------
        |        IntSlider         |
        ----------------------------

        """

        if 'logo' in kwargs:
            if 'logo_size' in kwargs:
                logo_size = kwargs.pop('logo_size')
            else:
                logo_size = ['auto', 'auto']
            imageWidget = ipywidgets.Image(
                value=open(kwargs.pop('logo'), 'rb').read(),
                format='png',
                width=logo_size[0],
                height=logo_size[1],
            )
            widgetStructure.insert(0,[imageWidget])

        # None to empty widgets
        empty = ipywidgets.Label(' ', layout=ipywidgets.Layout(visibility='hidden'))
        for horizontalWidgets in widgetStructure:
            for i in range(len(horizontalWidgets)):
                if horizontalWidgets[i] == None:
                    horizontalWidgets[i] = empty

        # apply layout to each widgets
        for w in sum(widgetStructure, []):
            if not w.layout.flex:
                w.layout.flex='1 1 auto'
            if not w.layout.width:
                w.layout.width = 'auto'

            if type(w) == ipywidgets.Button:
                w.add_class('pyButton')

        # row layout
        Hbox_layout = ipywidgets.Layout(
            display='flex',
            flex_flow='row',
            align_items='stretch'
        )
        # app box layout

        if width and (not type(width) == str and (not width.endswith('%') or not width.endswith('px'))):
            raise ValueError('\nUnsupported value in "width"\nexample)\n\twidth="240px"')

        Vbox_layout = ipywidgets.Layout(
            display='inline-flex',
            flex_flow='column wrap',
            align_items='stretch',
            width='auto' if width is None else str(width),
            padding='10px 10px 10px 10px',
            border='solid 2px lightgray'
        )

        # get rows
        if HboxLayout:
            verticalItems = [ipywidgets.Box(children=horizontalWidgets, layout=HboxLayout) for horizontalWidgets in
                             widgetStructure]
        else:
            verticalItems = [ipywidgets.Box(children=horizontalWidgets, layout=Hbox_layout) for horizontalWidgets in
                             widgetStructure]

        # make appLayout
        result = ipywidgets.VBox(verticalItems, layout=VboxLayout if VboxLayout else Vbox_layout)
        result.add_class('appLayout')
        if run:
            display(result)
        else:
            return result

    @staticmethod
    def styling(cssFile=None, cssStr=None):
        """
        apply css styling

        :param cssFile: css file path
        :param cssStr: css code string
        :return:
        """

        if not cssFile and not cssStr:
            print("One of 'cssFile','cssStr' parameter required.")
            return

        css = ""

        if cssFile:
            css = open(cssFile, 'r', encoding='utf8').read()

        if cssStr:
            css = cssStr

        return HTML(css)

    @staticmethod
    def messageBox(title, message):
        display(Javascript("""
            require(
                ["base/js/dialog"], 
                function(dialog) {{
                    dialog.modal({{
                        title: '{}',
                        body: '{}',
                        buttons: {{
                            '확인': {{}}
                        }}
                    }});
                }}
            );
        """.format(title, message)))

    @staticmethod
    def clear():
        clear_output()