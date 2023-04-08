import turtle
from winrt._winrt import initialize_with_window,create_direct3d11_device_from_dxgidevice, winrt_base
from winrt.windows.foundation import IAsyncOperation, AsyncStatus
from winrt.windows.graphics.capture import GraphicsCapturePicker, GraphicsCaptureItem
from winrt.windows.graphics.capture import GraphicsCaptureSession, Direct3D11CaptureFramePool, Direct3D11CaptureFrame
from winrt.windows.graphics.directx import DirectXPixelFormat
from winrt.windows.graphics.directx.direct3d11 import IDirect3DDevice, IDirect3DSurface
from winrt.windows.graphics import SizeInt32
from winrt.windows.foundation import  EventRegistrationToken
from winrt.windows.system import DispatcherQueue, DispatcherQueueController
from winrt.windows.graphics.imaging import SoftwareBitmap
from winrt.windows.storage.streams import IBuffer
from winrt.windows.security.cryptography import CryptographicBuffer
        
from typing import Optional
import asyncio
class Recorder:
    """ Capture Api """
    _lastSize: Optional[SizeInt32] = None
    _item : Optional[GraphicsCaptureItem] = None
    _framePool:Optional[Direct3D11CaptureFramePool] = None
    _session:Optional[GraphicsCaptureSession] = None
    """ non capture api """
    _canvasDevice:Optional[IDirect3DDevice] = None
    _current_frame = None
    
    def __init__(self):
        if(GraphicsCaptureSession.is_supported()):
            self._canvasDevice = IDirect3DDevice._from(create_direct3d11_device_from_dxgidevice())
        else:
            raise(Exception("not supported"))

    def __on_pick_completed(self, op: IAsyncOperation, status: AsyncStatus) -> None:
        if status == AsyncStatus.ERROR:
            print("error: ", status.error_code.value)
        elif status == AsyncStatus.CANCELED:
            # this is programatically, canceled, not user canceled
            print("operation canceled")
        elif status == AsyncStatus.COMPLETED:
            result: Optional[GraphicsCaptureItem] = op.get_results()
            if result:
                self._item = result
                print("result:",  result.display_name)
            else:
                print("user canceled")

        op.close()
        turtle.bye()

    def _start_capture_internal(self, item:GraphicsCaptureItem):
        print('_start_capture_internal')
        self._item = item
        self._lastSize = self._item.size
        print(self._lastSize.width,self._lastSize.height)
        self._framePool = Direct3D11CaptureFramePool.create(self._canvasDevice, DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED, 2, self._item.size)
        # self._framePool = Direct3D11CaptureFramePool.create_free_threaded(self._canvasDevice, DirectXPixelFormat.B8_G8_R8_A8_UINT_NORMALIZED, 2, self._item.size)
        # self._framePool.dispatcher_queue = DispatcherQueue.get_for_current_thread()
        self._framePool_token = self._framePool.add_frame_arrived( self.__on_frame_arrived)
        self._item_toke =  self._item.add_closed(self.__on_close)
        self._session = self._framePool.create_capture_session(self._item)
        self._session.start_capture()
        print("dispacherqueue",self._framePool.dispatcher_queue)
        
        dc = DispatcherQueueController.create_on_dedicated_thread()
        print("DispatcherQueueController",dc.dispatcher_queue)
        print('_start_capture_internal END')
        
    def __on_close(self)->None :
        print("__on_close")
        self.stop_capture()
        
    def stop_capture(self, s,a):
        print('stop_capture')
        # self._session.dispose()
        # self._framePool.dispose()
        self._item = None
        self._session = None
        self._framePool = None
        
    def __on_frame_arrived(self, s:Direct3D11CaptureFramePool,a:winrt_base )->None :
        print('__on_frame_arrived')
        # frame = self._framePool.try_get_next_frame()
        frame = s.try_get_next_frame()
        print(frame)
        # self._process_frame(frame)
        
    async def screenshot(self):
        print('screenshot')
        try:
            frame = self._framePool.try_get_next_frame()
            if frame:
                await self._process_frame(frame)
            else:
                print("pool is empty")
        except Exception as e:
            print(e)

    async def _process_frame(self, frame:Direct3D11CaptureFrame):
        print('_process_frame')
        need_reset:bool = False
        recreate_device: bool = False
        print(frame)
        if((frame.content_size.width != self._lastSize.width) or (frame.content_size.height != self._lastSize.height)):
            need_reset = True
            self._lastSize = frame.content_size
        print(f"""
                  ====================================
                  {frame.content_size.width} {frame.content_size.height}
                  ====================================
                  """)   
        print(frame.surface.description)
        try:
            await self.from_surface_to_buffer(frame.surface)
            pass
        except:
            need_reset = True
            recreate_device = True
            
        if need_reset:
            #reset_frame_pool(frame.content_size, recreate_device)
            pass
        
        exit()
        
    # async def from_surface_to_buffer(self, surface:IDirect3DSurface):
    #     print("from_surface_to_buffer")
        
    #     software_bitmap = await SoftwareBitmap.create_copy_from_surface_async(surface)
    #     print('1',software_bitmap)
    #     # ibuffer:IBuffer = IBuffer() # need implement IBuffer contructor
    #     software_bitmap.copy_to_buffer(ibuffer)
    #     print('1',ibuffer)
    #     mylist = CryptographicBuffer.copy_to_byte_array(ibuffer)
    #     print('LIST',mylist)
        
    async def from_surface_to_buffer(self, surface:IDirect3DSurface):
        print("from_surface_to_buffer")

        try:
            software_bitmap = await SoftwareBitmap.create_copy_from_surface_async(surface)
            print('1',software_bitmap)
            print('BitmapAlphaMode',software_bitmap.bitmap_alpha_mode)
            print('BitmapPixelFormat',software_bitmap.bitmap_pixel_format)
            print('DpiX',software_bitmap.dpi_x)
            print('DpiY',software_bitmap.dpi_y)
            print('PixelHeight',software_bitmap.pixel_height)
            height = software_bitmap.pixel_height
            print('PixelWidth',software_bitmap.pixel_width)
            width = software_bitmap.pixel_width
            ibuffer:IBuffer = CryptographicBuffer.generate_random(4294967295)
            print('Buffer',ibuffer.length)
            software_bitmap.copy_to_buffer(ibuffer)
            print('2',ibuffer,ibuffer.length)
            mylist = CryptographicBuffer.copy_to_byte_array(ibuffer)
            # with open('bitmap.py', 'w') as f:
            #     f.write('mylist = ' +mylist.__str__())
            import cv2
            import numpy as np
            im = np.array(mylist).reshape(height,width,4)

            cv2.imwrite('temp.png',im)
            s = cv2.imread('temp.png')
            cv2.imshow('capture result',s)
            cv2.waitKey(0) 
            cv2.destroyAllwindows() 
        except Exception as e:
            print(e)
        
    def reset_frame_pool(self, size:SizeInt32, recreate_device:bool):
        pass

    def _start_capture_async(self):
        print('_start_capture_async')
        picker = GraphicsCapturePicker()
        initialize_with_window(picker, turtle.getcanvas().winfo_id())


        op = picker.pick_single_item_async()
        op.completed = self.__on_pick_completed
        
        turtle.mainloop()
        if self._item != None:
            self._start_capture_internal(self._item)
            pass
        else:
            raise('not choose any window')
        
    def start(self):
        self._start_capture_async()
    
async def get(rec:Recorder):
    while True:
        await rec.screenshot()
        await asyncio.sleep(2)


if __name__ == '__main__':
    recorder = Recorder()
    recorder.start()
    asyncio.run(get(recorder))
    # recorder.screen_shot()