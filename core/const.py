# -*- coding: utf-8 -*-
"""
===============================================
const module
===============================================

========== ====================================
========== ====================================
 Module     const module
 Date       2019-04-20
 Author     hian
 Comment    `관련문서링크 <>`_
========== ====================================

*Abstract*
    *
    >>> EXAMPLE
    from common.core import const
    photobook_title = data[const.CLUSTER.PHOTOBOOKS.TITLE]



===============================================
"""


class _JSON_OBJECT(object):

    def __init__(self, object_name=None):
        if object_name is None:
            self._name = self.__class__.__name__.lower()
        else:
            self._name = object_name

    def __repr__(self):
        return repr(self._name)


# region internal objects
class IMAGES(_JSON_OBJECT):
    INDEX = 'index'
    IMAGEKEY = 'imageKey'
    IMAGE_ORI_FILE = 'imageOriFile'
    ORIPQ_W = 'oripqW'
    ORIPQ_H = 'oripqH'
    OT = 'ot'
    EXIF_DATE = 'exifDate'
    SYS_DATE = 'sysDate'
    GPS = 'gps'
    TYPE = 'type'


IMAGES_LEGNTH = 'imagesLength'
