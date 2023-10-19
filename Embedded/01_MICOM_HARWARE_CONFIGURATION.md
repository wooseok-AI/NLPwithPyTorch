01.MICOM_HARWARE_CONFIGURATION

# 임베디드 마이크로 컴퓨터 구성

임베디드 시스템의 하드웨어에서 공통된 부분을 소개하고, 임베디드 시스템에서 소프트웨어를 동작시키기 위한 필요 지식에 대해 설명한다

## 하드웨어의 종류
* CPU
  * Bus를 통해서 명령을 전달하고 입력을 전달 받는다. 
* Memory
  * ROM (Read Only Memory) 
  * RAM (Random Access Memory) 
* Peripheral(주변장치)

### 메모리의 종류

**(1) ROM (Read Only Memory)**

프로그램을 보관해 두고, CPU가 참조하여 동작을 수행한다. 제조공정에서 데이터를 써놓으면 바꿀 수 없는 MaskROM, 작성이 가능한 ProgrammableROM으로 나뉜다.

* ROM
  * MaskROM (명령코드, 상수 저장)
    * 제조공정에 데이터가 써지며, 삭제 불가함. 저렴
  * FlashMemory (명령코드, 상수 저장)
    * 전기적으로 저장하며, 재기록 가능하다. 
  * EEPROM (명령코드, 상수 저장)
    * 바이트 단위로 삭제 가능하며, 용량이 적다.
  * EPROM (명령코드, 상수)
    * 자외선을 통한 저장, 재기록 가능
  * PROM (OTP) 
    * 삭제 불가로 1회의 재기록 가능

**(2) RAM (Random Access Memory)**

데이터 보관과 무관한 기능을 전환할 때 사용되며, DynamicRAM (DRAM), StaticRAM(SRAM) 두 종류가 있다.

* RAM
  * SRAM 
    * 전기적으로 기록,삭제하며 소비전류가 적다.
  * DRAM
    * 프로그램처리, 고밀도

### 버스의 구성

메인버스와 로컬버스로 나뉘며, 메모리 등의 고속 제어장치는 메인버스를 통해 CPU와 접속하며, 이외의 주변장치는 Bridge를 경유하여 CPU와 통신함.

**메인버스**

각종 버스 신호는 Clock(신호)에 의해 동기화되어 디바이스(메모리, 주변장치)에 지시를 전달하는데에 사용된다.

(1) 주소 버스 : 특정 디바이스에 접근하기 위해 이용하는 신호

(2) 데이터 버스 : 디바이스로부터 데이터를 읽어들이기 위한 신호

(3) 컨트롤 버스 : 디바이스 제어를 위한 신호선

</br>

**로컬버스**

메인버스의 클럭 속도보다 저속으로 동작하는 디바이스를 제어하는 신호선

(1) 브릿지 

메인버스와 로컬버스를 연결하는 컨트롤러, FIFO룰 구현한 하드웨어 등에서, 저속의 로컬버스와 메인 버스 타이밍에 맞춰서 데이터를 송수신해주는 하드웨어

(2) UART

동기식 직렬신호를 병렬신호로, 혹은 역으로 변환하는 하드웨어. CPU로부터 8~16비트 폭으로 데이터가 병렬 전송되며, 이를 직렬 신호로 (Tx) 전송한다. 수신(Rx)에는 병렬로 변환 후에 송신한다.

UART 끼리 통신시에는 데이터의 시작과 긑을 데이터 사이에 신호로 보내어 송수신을 하며, 호스트 PC와 임베디드 시스템에 접속하여 시리얼 콘솔로 테스트나 디버깅할때 많이 사용된다.

(3) I2C

Serial Clock(SCL)과 Serial Data(SDA) 두개의 신호선을 사용하여 통신하는 동기식 직렬 통신. 마스터와 슬레이브가 있어서 복수 슬레이브에 연결이 가능하다. bit rate (초당 전송 비트수)는 표준(100Kbit/s), 저속(400Kbit/s), 고속(3.4Mbit/s)가 있다.

    **동기식/비동기식**

    동기와 비동기를 나누는 가장 큰 차이점을 어떻게 실행 순서를 가지는 지에 있다.

    Syncronous 동기는 요청을 보낸 후 해당 요청의 응답을 받아야 다음 동작을 실행하는 방식을,

    Asynchronous 비동기는 요청을 보낸 후 응답과 관계없이 다음 동작을 실행할 수 있는 방식을 의미한다.


(4) SPI

시리얼 커뮤니케이션을 하며, Serial Clock과 단방향 Seiral Data In (SDI), Serial Data Out (SDO)로 통신하는 동기식 직렬 통신. 복수의 슬레이브에 접속 가능하며, Slave Select를 이용하여 Slave 선택하여 통신. 많은 신호선이 필요하지만 I2C보다 빠르다.


### Peripheral (주변장치)

(1) DMA (Direct Memory Access) Controller

CPU가 PIO(Programmed I/O)를 통해 메모리를 읽고 쓸수있지만, 자원관리를 위해서 DMA를 통해 CPU를 사용하지 않고 데이터를 읽고 쓴다. Bus Aribiter를 이용해서 데이터 충돌이 없이 DMA 가 데이터 처리를 하도록 돕는다.

> 데이터 전송 도중에 버스를 점유하여 메모리 엑세스가 늦어질 수 있으므로, 전송 설정 (주소, 사이즈)를 잘 설정해야한다.

(2) Timer

Peripheral의 지속적인 감시 혹은 주기적 데이터 출력을 위한 시간 관련 처리에서 필수적이다. 카운터라고 불리는 레지스터에 주기 시간을 설정하고, 시간 경과하면 Interrupt발생. CPU는 해당 인터럽트를 통해서 프로그램을 동작시켜 주기적 처리를 실현함.

(3) RTC (Real Time Clock)

시간관리 장치. CPU 정지후 재개시 정확한 시간을 제공

(4) GPIO (General Perpose Input/Output)

CPU에 직접 연결되어 입력 및 출력을 범용으로 사용되는 포트. 외부의 주변장치로부터 인터럽트 신호에 사용되는 등 범용적으로 IO에 이용 가능하다.

### 제어방식

대다수 CPU의 Resigster라는 제어용 메모리를 사용해서 제어함. CPU 관점에서 레지스터 제어는 Memory mapped I/O와 I/O mapped I/O가 있다.

https://code-lab1.tistory.com/204

* Memory Mapped I/O : 

  ROM, RAM과 공통 주소를 이용하여 특정 주소에 대해 읽고 씀. ROM,RAM과 마찬가지로 주변장치의 레지스터도 메모리로 취급함.

* I/O Mapped I/O

  I/O 맵드 I/O의 경우, ROM, RAM는 메모리 공간으로 취급하고, 주변장치의 레지스터는 전용 명령으로 제어한다.

</br>

# CPU란

ROM으로 부터 실행해야할 절차를 읽어들이며, 읽어들인 절차를 해석해서 실행한다.
실행 결과는 RAM등에 보관한다.

<table style="border:1px solid black;margin-left:auto;margin-right:auto;">
  <tr>
    <th style="text-align: center; vertical-align: middle;" colspan="2"><b>CPU 구조</b></th>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;" colspan="2">Program Counter</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;" colspan="2">Decoder</th>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;">System Register</td>
    <td style="text-align: center; vertical-align: middle;">General Purpose Register</td>
  </tr>
  <tr>
    <td style="text-align: center; vertical-align: middle;" colspan="2">Arithmetic and Logic Unit</td>
  </tr>
</table>

* Progrm Counter

프로그램 카운터는 ROM내의 프로그램의 어디를 참고할지 관리함. Program Counter는 CPU 명령의 보관장소를 관리하여, 다음에 실행해야할 명령을 읽을 위치를 CPU에 전달하는 역할을 함.

* Decoder

불러온 명령을 해독하여 ALU에서의 연산이나, 데이터의 이동등 구체적 지시를 CPU내에서 실행한다.

* ALU

정수의 사칙연산 혹은 AND, OR, NOT 등의 논리연산을 실행함. 
ALU에서 연산을 수행한 후에는 범용 레지스터나 시스템 레지스터에 반영한다.

* General Purpose Register (범용레지스터)

CPU에 내장된 범용 메모리로, 고속이지만 용량이 작아 일시적으로 이용된다.  ALU의 연산결과를 보관하거나 데이터를 이동할 떄의 보관장소로 이용된다.

* System Register (시스템 레지스터)

CPU 명령 실행을 위해 이용되며, 명령을 보관하는 명령 레지스터, 주소 관리용 주소 레지스터, CPU상태 관리용 플래그 레지스터 등이 있다.

|7bit|6bit|5bit|4bit|3bit|2bit|1bit|0bit|
|---|---|---|---|---|---|---|---|
|I|-|H|S|-|N|Z|C|

|비트|이름|용도|
|---|---|---|
|0|C : Carry Flag| ALU에서의 연산결과로 OverFlow가 발생한 상태관리|
|1|Z : Zero Flag| ALU에서의 연산 결과가 0이 된 상태를 관리|
|2|N : Negative Flag| ALU애서의 연산이 덧셈인지 뺄셈인지 상태관리|
|4|S : Sign Flag| ALU에서의 연산결과가 음수가 되었는지의 상태관리|
|5|H : Half Carry Flag| ALU 에서의 연산결과로 Overflow된 상태관리, Carry Flag가 최상위 비트를 보고 있는데 반해, 중간 3비트째의 오버플로를 관리한다. Binary Coded Decial의 연산에 이용|
|7|I : Interrupt Eable| 인터럽트 허가하는지에 대한 상태관리|

## CPU 명령실행

1. 명령 패치 사이클 : ROM에서 명령 추출
2. 명령 디코드 사이클 : 추출한 명령 해독 및 실행준비
3. 실행 사이클 : 명령 실행 
4. 라이트 백 사이클 : 명령 실행 결과의 반영

## CPU 명령의 종류

데이터 취급 명령의 대부분은 범용 레지스터를 활용함. 범용 레지스터의 데이터를 메모리에 기록하거나, 메모리에 있는 데이터를 범용 레지스터로 읽어 들인다.

* CPU와 메모리 사이의 데이터 교환명령
* CPU와 주변장치 사이의 데이터 교환명령
* CPU 내부명령

## 인터럽트

주변장치로 부터의 처리요구를 CPU에 통지하기 위한 신호이다.
인터럽트는 interrupt vectro table에 저장되며, 미리 정해진 프로그램에 맞게 점프한다. 인터럽트 처리 종료후에는 일상적 동작을 재개한다.

### 인터럽트의 종류

* 타이머 인터럽트

  소정의 시간이 되면 인터럽트를 발생시키는 의도를 가지고 사용함

* 외부 인터럽트

  주변장치의 상태에 따른 인터럽트

인터럽트의 종류에는 타이머 처리, DMA 처리, 스위치 처리, 렌더링 처리, 시리얼 처리 가 있다. 인터럽트 벡터에는 처리 그 자체를 등록하는 것이 아니라 메모리 주소를 저장한다.

### 인터럽트 우선순위

인터럽트 처릴 순서를 위해서 우선순위를 정해야한다. 인터럽트의 우선순위를 지정해두고, 인터럽트 벡터에 프로그램 번지를 등록한다. 특정 CPU에는 우선순위를 위한 레지스터가 있다.


