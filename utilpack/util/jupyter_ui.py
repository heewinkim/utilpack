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

    @staticmethod
    def select(dataList, options, runFunc,initFunc=None,backFunc=None,width=None, description='데이터 리스트',**funcKwargs):
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
            runFunc(selected_data,initFunc=initFunc,backFunc=backFunc **funcKwargs)

        runButton.on_click(onClickRun)
        initButton.on_click(lambda v:initFunc())
        backButton.on_click(lambda v:backFunc())

        widgetStructure = [
            [list_widget],
            [runButton]
        ]
        if backFunc:
            widgetStructure[-1].append(backButton)
        if initFunc:
            widgetStructure[-1].append(initButton)

        PyUI.appLayout(widgetStructure,width=width)

    @staticmethod
    def back(initFunc=None,backFunc=None,clearOutput=True,**kwargs):
        """
        init,back 버튼 및 버튼 콜백 함수를 정의합니다.

        :param initFunc: initFunc(**kwargs['init'])
        :param backFunc: backFunc(**kwargs['init'])
        :param clearOutput: boolean
        :param kwargs: kwargs, consist of 'init','back',
        :return: PyUI.ButtonController(initButton,backButton)
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
        HTML("""
        <style>
        .pyButton {
            border-radius: 5px;
            color:rgb(255,255,255) ; 
            background-color:rgb(70,70,70);
            font-size:100%
        }
        </style>
        """)

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
        widget_layout = ipywidgets.Layout(flex='1 1 auto', width='auto')
        for w in sum(widgetStructure, []):
            w.layout = widget_layout
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
        if run:
            display(result)
        else:
            return result

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