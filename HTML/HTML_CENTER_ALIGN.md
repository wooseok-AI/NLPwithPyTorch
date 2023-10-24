HTML_CENTER_ALIGN

# 가운데 정렬

table 는 조직적인 방식으로 많은 정보를 표시하는 훌륭한 방법입니다 . 판매 데이터, 웹 페이지 트래픽, 주식 시장 동향 및 학생의 성적은 종종 table 에 표시되는 정보의 예입니다.



HTML 을 사용하여 웹 페이지에 table 을 추가할 때 페이지 중앙에 table 을 배치하는 것이 시각적으로 더 매력적일 수 있습니다. 텍스트와 그림을 가운데에 맞추는 작업은 일반적으로 text-align 클래스나 CSS 를 통해 이루어 지지만, table을 가운데에 맞추려면 다른 접근 방식이 필요합니다. 웹 페이지에서 표를 가운데에 맞추는 방법에 대한 자세한 내용은 아래에 나와 있습니다.

## HTML에서 Table 중앙에 맞추기

웹 페이지에 table 을 추가할 때 기본적으로 아래와 같이 페이지 또는 컨테이너의 왼쪽에 정렬됩니다.

<table style="border:1px 단색 검정">
  <tr>
    <td><b>히트</b></td>
    <td><b>MONTH</b></td>
    <td><b>총 증가</b></td>
  </tr>
  <tr>
    <td>324,497</td>
    <td>1998년 1월 </td>
    <td style="text-align:center">-</td>
  </tr>
    <tr>
    <td>436,699</td>
    <td>1998년 2월</td>
    <td style="text-align:center">112,172</td>
  </tr>
</table>

위 표의 HTML 소스 코드 는 다음과 같습니다.

```html
<table style="border:1px 단색 검정">
  <tr>
    <td><b>히트</b></td>
    <td><b>MONTH</b></td>
    <td><b>총 증가</b></td>
  </tr>
  <tr>
    <td>324,497</td>
    <td>1998년 1월 </td>
    <td style="text-align:center">-</td>
  </tr>
    <tr>
    <td>436,699</td>
    <td>1998년 2월</td>
    <td style="text-align:center">112,172</td>
  </tr>
</table>
```

이 table 을 가운데에 맞추려면 
```html
;margin-left:auto;margin-right:auto;
```
table 태그 의 style 속성 끝에 table 태그는 다음과 같습니다.

```html
<table style="border:1px solid black;margin-left:auto;margin-right:auto;">
```
 
위와 같이 table 태그의 style 속성을 변경하면 아래와 같이 table 이 웹 페이지의 중앙에 위치하게 됩니다.

<table style="border:1px solid black;margin-left:auto;margin-right:auto;" style="border:1px 단색 검정">
  <tr>
    <td><b>히트</b></td>
    <td><b>MONTH</b></td>
    <td><b>총 증가</b></td>
  </tr>
  <tr>
    <td>324,497</td>
    <td>1998년 1월 </td>
    <td style="text-align:center">-</td>
  </tr>
    <tr>
    <td>436,699</td>
    <td>1998년 2월</td>
    <td style="text-align:center">112,172</td>
  </tr>
</table>
